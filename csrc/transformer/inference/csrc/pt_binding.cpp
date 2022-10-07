/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "inference_context.h"
#include "inference_cublas_wrappers.h"
#include "inference_cuda_layers.h"

std::array<int, 3> gemm_algos = std::array<int, 3>({99, 99, 99});

// NOTE: This activation function type enum should be always in sync
// with the python counterpart, otherwise the casting from python binding
// will be incorrect.
enum class ActivationFuncType { UNKNOWN = 0, GELU = 1, ReLU = 2 };

enum class TransformerType : uint8_t { UNKNOWN = 0, GPTType = 1, BERTType = 2 };

// NOTE: this is a temporary and dodgy solution to distinguish GPT and BERT style models
// based on the dimensions of the corresponding attention mask.
inline auto infer_transformer_type(at::Tensor& attn_mask) -> TransformerType
{
    auto attn_mask_num_dims = attn_mask.sizes().size();

    if (attn_mask_num_dims > 2) {
        return TransformerType::GPTType;
    } else if (attn_mask_num_dims == 2) {
        return TransformerType::BERTType;
    } else {
        return TransformerType::UNKNOWN;
    }
}

// infer stride of attention mask memory layout based on the model type.
inline auto get_attn_mask_stride(at::Tensor& attn_mask) -> int
{
    auto trnsfrmr_type = infer_transformer_type(attn_mask);

    if (trnsfrmr_type == TransformerType::GPTType) {
        return attn_mask.size(2);
    } else if (trnsfrmr_type == TransformerType::BERTType) {
        // Bert style models have always a mask stride of 1.
        return 1;
    } else if (trnsfrmr_type == TransformerType::UNKNOWN) {
        return 0;
    }

    // this is just to make the compiler happy.
    return 0;
}

template <typename T>
at::Tensor ds_softmax(at::Tensor& attn_scores,
                      at::Tensor& attn_mask,
                      at::Tensor& alibi,
                      bool triangular,
                      bool recompute,
                      bool local_attention,
                      int window_size,
                      bool async_op,
                      float layer_scale,
                      int head_offset,
                      int mp_size)
{
    auto attn_scores_c = attn_scores.contiguous();
    int bsz = attn_scores_c.size(0);

    int seq_len = attn_scores_c.size(1);
    int len = attn_scores_c.sizes().size();
    if (len > 2) seq_len = attn_scores_c.size(2);

    int soft_len = attn_scores_c.size(2);
    if (len > 3) soft_len = attn_scores_c.size(3);

    int heads = 1;
    if (len > 1) heads = attn_scores_c.size(1);

    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores_c.data_ptr(),
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           head_offset,
                           mask_stride,
                           mp_size,
                           Context::Instance().GetCurrentStream(async_op));

    return attn_scores_c;
}

template <typename T>
void allocate_workspace(size_t hidden_dim,
                        size_t batch_size,
                        unsigned num_layers,
                        unsigned mp_size = 1,
                        unsigned rank = 0)
{
    Context::Instance().GenWorkSpace(num_layers, batch_size, hidden_dim, mp_size, sizeof(T), rank);
}

template <typename T>
at::Tensor einsum_sec_sm_ecm(at::Tensor& Q, at::Tensor& W)
{
    auto options = at::TensorOptions()
                       .dtype(Q.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    float alpha = 1;
    float gemm_beta = 0.0;

    auto O = at::from_blob(workspace, {Q.size(1), Q.size(2), W.size(1)}, options);
    unsigned m = W.size(1);
    unsigned n = Q.size(1) * Q.size(2);
    unsigned k = Q.size(0);
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_N,
                   CUBLAS_OP_T,
                   m,
                   n,
                   k,
                   &alpha,
                   &gemm_beta,
                   (T*)W.data_ptr(),
                   (T*)Q.data_ptr(),
                   (T*)O.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    return O;
}

template <typename T>
void ds_softmax_internal(T* attn_scores,
                         at::Tensor& attn_mask,
                         at::Tensor& alibi,
                         float& layer_scale,
                         bool triangular,
                         bool recompute,
                         bool local_attention,
                         int window_size,
                         int bsz,
                         int seq_len,
                         int soft_len,
                         int heads,
                         int alibi_offset,
                         int mp_size)
{
    auto mask_stride = get_attn_mask_stride(attn_mask);

    launch_attn_softmax_v2((T*)attn_scores,
                           (attn_mask.sizes().size() > 1 ? (T*)attn_mask.data_ptr() : nullptr),
                           (alibi.sizes().size() > 1 ? (T*)alibi.data_ptr() : nullptr),
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           heads,
                           seq_len,
                           soft_len,
                           alibi_offset,
                           mask_stride,
                           mp_size,
                           at::cuda::getCurrentCUDAStream());
}

template <typename T>
void attention_unfused(T* prev_key_cont,
                       T* query_cont,
                       at::Tensor& attn_mask,
                       T* prev_value_cont,
                       T* output,
                       unsigned& bsz,
                       int& k,
                       unsigned& seq_len,
                       unsigned& soft_len,
                       int& heads,
                       float& norm_factor,
                       bool triangular,
                       bool recompute,
                       bool local_attention,
                       int window_size,
                       at::Tensor& alibi,
                       int alibi_offset,
                       int mp_size,
                       int layer_id)
{
    float layer_scale = alibi.sizes().size() > 1 ? std::max(1, layer_id) : 1.0;
    float alpha = norm_factor * norm_factor / layer_scale;
    float gemm_beta = 0.0;
    T* workspace;

    // If we are doing the prompt, switch to the tail workspace
    T* scratch = (T*)Context::Instance().GetWorkSpace();
    workspace = scratch + ((Context::Instance().get_workspace_size() / sizeof(T)) -
                           bsz * heads * seq_len * soft_len);

    cublasSetStream(Context::Instance().GetCublasHandle(), Context::Instance().GetCurrentStream());
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                soft_len,
                                seq_len,
                                k,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_key_cont,
                                (T*)query_cont,
                                workspace,
                                CUBLAS_OP_T,
                                CUBLAS_OP_N,
                                Context::Instance().GetMaxTokenLenght() * k,
                                seq_len * k,
                                seq_len * soft_len,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
    ds_softmax_internal<T>(workspace,
                           attn_mask,
                           alibi,
                           layer_scale,
                           triangular,
                           recompute,
                           local_attention,
                           window_size,
                           bsz,
                           seq_len,
                           soft_len,
                           heads,
                           alibi_offset,
                           mp_size);
    alpha = 1.0;
    cublas_strided_batched_gemm(Context::Instance().GetCublasHandle(),
                                k,
                                seq_len,
                                soft_len,
                                &alpha,
                                &gemm_beta,
                                (T*)prev_value_cont,
                                workspace,
                                (T*)output,
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                Context::Instance().GetMaxTokenLenght() * k,
                                seq_len * soft_len,
                                seq_len * k,
                                bsz * heads,
#ifdef __HIP_PLATFORM_HCC__
                                rocblas_gemm_algo_standard);
#else
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
std::vector<at::Tensor> ds_softmax_context(at::Tensor& query_key_value,
                                           at::Tensor& attn_mask,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           int heads,
                                           float norm_factor,
                                           bool triangular,
                                           bool local_attention,
                                           int window_size,
                                           bool no_masking,
                                           unsigned layer_id,
                                           unsigned num_layers,
                                           at::Tensor& alibi,
                                           int alibi_offset,
                                           int mp_size)
{
    unsigned bsz = query_key_value.size(0);
    unsigned seq_len = query_key_value.size(1);
    unsigned hidden_dim = query_key_value.size(2) / 3;

    bool is_prompt = (seq_len > 1);

    if (is_prompt) Context::Instance().reset_tokens(seq_len);
    unsigned soft_len = Context::Instance().current_tokens();

    int k = hidden_dim / heads;
    auto options = at::TensorOptions()
                       .dtype(query_key_value.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* workspace = (T*)Context::Instance().GetWorkSpace();
    size_t buf_size = bsz * seq_len * hidden_dim * mp_size;
    auto output = torch::from_blob(workspace + 4 * buf_size, {bsz, seq_len, hidden_dim}, options);

    auto query_cont = workspace + 8 * buf_size;
    size_t offset = 16 * (hidden_dim * mp_size * bsz * Context::Instance().GetMaxTokenLenght()) +
                    layer_id * 2 * bsz * Context::Instance().GetMaxTokenLenght() * hidden_dim;
    unsigned all_tokens = soft_len;
    auto kv_cache = workspace + offset + (hidden_dim / heads) * (is_prompt ? 0 : soft_len - 1);
    size_t value_offset = bsz * Context::Instance().GetMaxTokenLenght() * hidden_dim;

    T* temp_buf = (T*)output.data_ptr() + at::numel(output);
    launch_bias_add_transform_0213<T>((T*)query_cont,
                                      kv_cache,
                                      kv_cache + value_offset,
                                      (T*)query_key_value.data_ptr(),
                                      nullptr,
                                      bsz,
                                      seq_len,
                                      (is_prompt ? 0 : soft_len - 1),
                                      soft_len,
                                      hidden_dim,
                                      heads,
                                      rotary_dim,
                                      rotate_half,
                                      rotate_every_two,
                                      Context::Instance().GetCurrentStream(),
                                      3,
                                      Context::Instance().GetMaxTokenLenght());
    if (rotary_dim > 0 && rotate_half)
        launch_apply_rotary_pos_emb(query_cont,
                                    kv_cache,
                                    k,
                                    seq_len,
                                    rotary_dim,
                                    (is_prompt ? 0 : soft_len - 1),
                                    heads,
                                    bsz,
                                    rotate_half,
                                    rotate_every_two,
                                    Context::Instance().GetCurrentStream(),
                                    Context::Instance().GetMaxTokenLenght());

    attention_unfused<T>(workspace + offset,
                         (T*)query_cont,
                         attn_mask,
                         workspace + offset + value_offset,
                         temp_buf,
                         bsz,
                         k,
                         seq_len,
                         all_tokens,
                         heads,
                         norm_factor,
                         (triangular && is_prompt),
                         is_prompt,
                         local_attention,
                         window_size,
                         alibi,
                         alibi_offset,
                         mp_size,
                         layer_id);
    launch_transform4d_0213<T>((T*)output.data_ptr(),
                               temp_buf,
                               bsz,
                               heads,
                               seq_len,
                               output.size(2),
                               Context::Instance().GetCurrentStream(false),
                               1);

    if (layer_id == num_layers - 1) Context::Instance().advance_tokens();
    auto prev_key = torch::from_blob(workspace + offset, {bsz, heads, all_tokens, k}, options);
    auto prev_value =
        torch::from_blob(workspace + offset + value_offset, {bsz, heads, all_tokens, k}, options);
    return {output, prev_key, prev_value};
}

template <typename T>
at::Tensor ds_bias_gelu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_gelu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_relu(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int intermediate_size = input_cont.size(2);

    launch_bias_relu((T*)input_cont.data_ptr(),
                     (T*)bias.data_ptr(),
                     intermediate_size,
                     bsz,
                     Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_add(at::Tensor& input, at::Tensor& bias)
{
    auto input_cont = input.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    int hidden_size = input_cont.size(2);

    launch_bias_add((T*)input_cont.data_ptr(),
                    (T*)bias.data_ptr(),
                    hidden_size,
                    bsz,
                    Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_bias_residual(at::Tensor& input, at::Tensor& residual, at::Tensor& bias)
{
    auto input_cont = input.contiguous();
    auto residual_cont = residual.contiguous();

    int bsz = input_cont.size(0) * input_cont.size(1);
    // launch_bias_residual((T*)input_cont.data_ptr(),
    //                      (T*)residual_cont.data_ptr(),
    //                      (T*)bias.data_ptr(),
    //                      bsz,
    //                      input_cont.size(2),
    //                      (bias.size(0) > 1),
    //                      Context::Instance().GetCurrentStream());
    return input_cont;
}

template <typename T>
at::Tensor ds_layernorm(at::Tensor& input_cont, at::Tensor& gamma, at::Tensor& betta, float epsilon)
{
    int bsz = input_cont.size(0) * input_cont.size(1);
    auto inp_norm = at::empty_like(input_cont);
    launch_layer_norm((T*)inp_norm.data_ptr(),
                      (T*)input_cont.data_ptr(),
                      (T*)gamma.data_ptr(),
                      (T*)betta.data_ptr(),
                      epsilon,
                      bsz,
                      input_cont.size(2),
                      Context::Instance().GetCurrentStream());
    return inp_norm;
}

template <typename T>
void ds_layernorm_internal(T* workspace,
                           at::Tensor& input,
                           at::Tensor& gamma,
                           at::Tensor& betta,
                           float epsilon)
{
    int bsz = input.size(0) * input.size(1);
    launch_layer_norm(workspace,
                      (T*)input.data_ptr(),
                      (T*)gamma.data_ptr(),
                      (T*)betta.data_ptr(),
                      epsilon,
                      bsz,
                      input.size(2),
                      Context::Instance().GetCurrentStream());
}

template <typename T>
void quantized_gemm(void* output,
                    T* input,
                    at::Tensor& weight,
                    at::Tensor& qscale,
                    int groups,
                    int bsz)
{
    T* weight16 = (T*)Context::Instance().GetWorkSpace() +
                  12 * Context::Instance().GetMaxTokenLenght() * weight.size(1);

    launch_dequantize(weight16,
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   weight.size(1),
                   &alpha,
                   &gemm_beta,
                   weight16,
                   (T*)input,
                   (T*)output,
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
at::Tensor qkv_unfused_cublas(at::Tensor& output,
                              at::Tensor& input,
                              at::Tensor& weight,
                              at::Tensor& q_scale,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool add_bias,
                              bool q_int,
                              unsigned q_bits)
{
    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    workspace += (3 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2));
    ds_layernorm_internal<T>(workspace, input, gamma, beta, epsilon);
    if (q_int) {
        int out_size = weight.size(0);
        int bsz1 = bsz;
        if (q_bits == 4 && 0) {
            // 128-aligned
            bsz1 = bsz % 128 == 0 ? bsz : (bsz / 128 + 1) * 128;
        } else {
            bsz1 = (bsz >= 32 && bsz < 128)

        int bsz1 = (bsz >= 32 && bsz < 128)
                       ? 128
                       : (bsz % 128 == 0)
                             ? bsz
                             : ((128 - (bsz % 128)) > 32 && bsz < 512)
                                   ? ((bsz % 64 == 0)
                                          ? bsz
                                          : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                : bsz + (64 - (bsz % 64)))
                                   : bsz + (128 - (bsz % 128));
        }
        auto aux_buff = (T*)Context::Instance().GetWorkSpace() +
                        8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);

        if (q_bits == 8 or q_bits == 4) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)workspace,
                      input.size(2),
                      bsz,
                      Context::Instance().GetCurrentStream());

            run_gemm(aux_buff,
                     weight.data_ptr(),
                     output.data_ptr(),
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            assert(q_bits == 4);
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)workspace,
                              input.size(2),
                              bsz,
                              Context::Instance().GetCurrentStream());
            run_gemm_int4(aux_buff,
                          weight.data_ptr(),
                          output.data_ptr(),
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2) * 2,
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }
        if (add_bias)
            launch_bias_add((T*)output.data_ptr(),
                            (T*)bias.data_ptr(),
                            out_size,
                            bsz,
                            Context::Instance().GetCurrentStream());
    } else {
        if (bsz > 1) {
            cublasSetStream(Context::Instance().GetCublasHandle(),
                            Context::Instance().GetCurrentStream());
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight.size(1),
                           bsz,
                           input.size(2),
                           &alpha,
                           &gemm_beta,
                           (T*)weight.data_ptr(),
                           workspace,
                           (T*)output.data_ptr(),
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            if (add_bias) {
                launch_bias_add((T*)output.data_ptr(),
                                (T*)bias.data_ptr(),
                                weight.size(1),
                                bsz,
                                Context::Instance().GetCurrentStream());
            }
        } else {
            launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                           workspace,
                                           (T*)weight.data_ptr(),
                                           (T*)(add_bias ? bias.data_ptr() : nullptr),
                                           input.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream());
        }
    }
    return torch::from_blob(workspace, input.sizes(), input.options());
}

template <typename T>
std::vector<at::Tensor> ds_qkv_gemm(at::Tensor& input,
                                    at::Tensor& weight,
                                    at::Tensor& q_scale,
                                    at::Tensor& bias,
                                    at::Tensor& gamma,
                                    at::Tensor& beta,
                                    const float epsilon,
                                    bool add_bias,
                                    unsigned num_layers,
                                    unsigned mp_size,
                                    unsigned rank,
                                    bool q_int,
                                    unsigned q_bits)
{
    int bsz = input.size(0) * input.size(1);
    int out_size = q_int ? weight.size(0) : weight.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    if (!workspace)
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());
    allocate_workspace<T>(input.size(2), input.size(0), num_layers, mp_size, rank);
    workspace = (T*)Context::Instance().GetWorkSpace();

    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    auto output =
        torch::from_blob(workspace, {input.size(0), input.size(1), out_size}, input.options());

    auto inp_norm = qkv_unfused_cublas<T>(
        output, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int, q_bits);
    return {output, inp_norm};
}

template <typename T>
void quantized_gemm(at::Tensor& output,
                    at::Tensor& input,
                    at::Tensor& weight,
                    at::Tensor& qscale,
                    int groups,
                    int merge_count)
{
    int bsz = input.size(0) * input.size(1);
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    auto weight16 = at::empty({weight.size(0), weight.size(1)}, options);

    launch_dequantize((T*)weight16.data_ptr(),
                      (int8_t*)weight.data_ptr(),
                      (float*)qscale.data_ptr(),
                      weight.size(0),
                      weight.size(1),
                      groups,
                      merge_count,
                      Context::Instance().GetCurrentStream());

    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                   CUBLAS_OP_T,
                   CUBLAS_OP_N,
                   weight.size(0),
                   bsz,
                   input.size(2),
                   &alpha,
                   &gemm_beta,
                   (T*)weight16.data_ptr(),
                   (T*)input.data_ptr(),
                   (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                   rocblas_gemm_algo_standard);
#else
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
}

template <typename T>
at::Tensor ds_linear_layer(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& q_scale,
                           at::Tensor& bias,
                           unsigned num_layers,
                           bool external_cache,
                           unsigned mp_size,
                           bool q_int,
                           unsigned q_bits)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    if (!workspace)
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());

    int out_size = q_int ? weight.size(0) : weight.size(1);
    allocate_workspace<T>(input.size(2), input.size(0), num_layers, mp_size, external_cache);
    workspace = (T*)Context::Instance().GetWorkSpace();
    auto output = at::from_blob(workspace, {input.size(0), input.size(1), weight.size(1)}, options);

    int bsz = input.size(0) * input.size(1);
    if (q_int) {
        int bsz1 = bsz;
        if (q_bits == 4 && 0) {
            // 128-aligned
            bsz1 = bsz % 128 == 0 ? bsz : (bsz / 128 + 1) * 128;
        } else {
            bsz1 = (bsz >= 32 && bsz < 128)
                       ? 128
                       : (bsz % 128 == 0)
                             ? bsz
                             : ((128 - (bsz % 128)) > 32 && bsz < 512)
                                   ? ((bsz % 64 == 0)
                                          ? bsz
                                          : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                : bsz + (64 - (bsz % 64)))
                                   : bsz + (128 - (bsz % 128));
        }
        auto aux_buff = (T*)Context::Instance().GetWorkSpace() +
                        8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);

        if (q_bits == 8 or q_bits == 4) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)input.data_ptr(),
                      input.size(2),
                      bsz,
                      Context::Instance().GetCurrentStream());

            run_gemm(aux_buff,
                     weight.data_ptr(),
                     output.data_ptr(),
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            assert(q_bits == 4);
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)input.data_ptr(),
                              input.size(2),
                              bsz,
                              Context::Instance().GetCurrentStream());

            run_gemm_int4(aux_buff,
                          weight.data_ptr(),
                          output.data_ptr(),
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }

        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        out_size,
                        bsz,
                        Context::Instance().GetCurrentStream());
    } else {
        float alpha = (T)1.0;
        float gemm_beta = (T)0.0;
        cublasSetStream(Context::Instance().GetCublasHandle(),
                        Context::Instance().GetCurrentStream());

        cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       weight.size(1),
                       bsz,
                       input.size(2),
                       &alpha,
                       &gemm_beta,
                       (T*)weight.data_ptr(),
                       (T*)input.data_ptr(),
                       (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                       rocblas_gemm_algo_standard);
#else
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif

        launch_bias_add((T*)output.data_ptr(),
                        (T*)bias.data_ptr(),
                        weight.size(1),
                        bsz,
                        Context::Instance().GetCurrentStream());
    }
    return output;
}
template <typename T>
at::Tensor ds_vector_matmul(at::Tensor& input,
                            at::Tensor& weight,
                            bool async_op,
                            at::Tensor& q_scale,
                            bool q_int,
                            unsigned q_bits)
{
    int out_size = q_int ? weight.size(0) : weight.size(1);
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);
    T* workspace = (T*)Context::Instance().GetWorkSpace() +
                   (5 * input.size(0) * Context::Instance().GetMaxTokenLenght() * out_size);

    auto output = torch::from_blob(workspace, {input.size(0), input.size(1), out_size}, options);
    int bsz = input.size(0) * input.size(1);
    if (q_int) {
        int bsz1;
        if (q_bits == 4 && 0) {
            // 128-aligned
            bsz1 = bsz % 128 == 0 ? bsz : (bsz / 128 + 1) * 128;
        } else {
            bsz1 = (bsz >= 32 && bsz < 128)
                       ? 128
                       : (bsz % 128 == 0)
                             ? bsz
                             : ((128 - (bsz % 128)) > 32 && bsz < 512)
                                   ? ((bsz % 64 == 0)
                                          ? bsz
                                          : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                : bsz + (64 - (bsz % 64)))
                                   : bsz + (128 - (bsz % 128));
        }
        auto aux_buff = (T*)Context::Instance().GetWorkSpace() +
                        8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * out_size;
        if (q_bits == 8 or q_bits == 4) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)input.data_ptr(),
                      input.size(2),
                      bsz,
                      Context::Instance().GetCurrentStream());

            run_gemm(aux_buff,
                     weight.data_ptr(),
                     workspace,
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)input.data_ptr(),
                              input.size(2),
                              bsz,
                              Context::Instance().GetCurrentStream());

            auto output = at::from_blob(workspace, input.sizes(), input.options());
            run_gemm_int4(aux_buff,
                          weight.data_ptr(),
                          workspace,
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        if (bsz > 1) {
            float alpha = (T)1.0;
            float gemm_beta = (T)0.0;
            cublasSetStream(Context::Instance().GetCublasHandle(),
                            Context::Instance().GetCurrentStream(async_op));
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight.size(1),
                           bsz,
                           input.size(2),
                           &alpha,
                           &gemm_beta,
                           (T*)weight.data_ptr(),
                           (T*)input.data_ptr(),
                           workspace,
#ifdef __HIP_PLATFORM_HCC__
                           rocblas_gemm_algo_standard);
#else
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
        } else {
            launch_input_tiled_gemm_kernel((T*)workspace,
                                           (T*)input.data_ptr(),
                                           (T*)weight.data_ptr(),
                                           (T*)nullptr,
                                           input.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream());
        }
    }
    return output;
}

at::Tensor act_quantized_gemm(at::Tensor& inp, at::Tensor& weight, at::Tensor q_scale)
{
    int bsz = inp.size(0) * inp.size(1);
    int bsz1 =
        (bsz >= 32 && bsz < 128)
            ? 128
            : (bsz % 128 == 0)
                  ? bsz
                  : ((128 - (bsz % 128)) > 32 && bsz < 512)
                        ? ((bsz % 64 == 0) ? bsz
                                           : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                 ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                 : bsz + (64 - (bsz % 64)))
                        : bsz + (128 - (bsz % 128));
    at::Tensor out = at::empty({bsz1, weight.size(0)}, inp.options());
    auto auxilary_buf = (__half*)Context::Instance().GetWorkSpace() +
                        ((Context::Instance().get_workspace_size() / sizeof(__half)) -
                         (bsz1 * inp.size(2) + bsz1 * inp.size(2) * 4));
    launch_me((int8_t*)auxilary_buf,
              (float*)((int8_t*)auxilary_buf + bsz1 * inp.size(2)),
              (__half*)inp.data_ptr(),
              inp.size(2),
              bsz,
              Context::Instance().GetCurrentStream());
    run_gemm(auxilary_buf,
             weight.data_ptr(),
             out.data_ptr(),
             (float*)((int8_t*)auxilary_buf + bsz1 * inp.size(2)),
             q_scale.data_ptr(),
             bsz1,
             weight.size(0),
             inp.size(2),
             bsz1,
             q_scale.size(0),
             Context::Instance().GetCurrentStream());
    return torch::narrow(out, 0, 0, bsz).view({inp.size(0), inp.size(1), weight.size(0)});
}

template <typename T>
at::Tensor mlp_unfused_cublas(T* output,
                              T* output2,
                              at::Tensor& input,
                              at::Tensor& residual,
                              at::Tensor& input_bias,
                              at::Tensor& weight,
                              at::Tensor& weight1,
                              at::Tensor& bias,
                              at::Tensor& gamma,
                              at::Tensor& beta,
                              const float epsilon,
                              bool preLayerNorm,
                              bool mlp_after_attn,
                              at::Tensor& q_scale,
                              at::Tensor& q_scale1,
                              bool q_int,
                              unsigned q_bits,
                              ActivationFuncType act_func_type)
{
    int bsz = input.size(0) * input.size(1);
    T* workspace = (T*)Context::Instance().GetWorkSpace() + 4 * at::numel(input);

    launch_residual_layer_norm(workspace,
                               (T*)nullptr,
                               (T*)input.data_ptr(),
                               (T*)residual.data_ptr(),
                               (T*)input_bias.data_ptr(),
                               (T*)gamma.data_ptr(),
                               (T*)beta.data_ptr(),
                               epsilon,
                               bsz,
                               input.size(2),
                               preLayerNorm,
                               mlp_after_attn,
                               Context::Instance().GetCurrentStream());
    if (q_int) {
        int out_size = weight.size(0);
        int bsz1;
        if (q_bits == 4 && 0) {
            // 128-aligned
            bsz1 = bsz % 128 == 0 ? bsz : (bsz / 128 + 1) * 128;
        } else {
            bsz1 = (bsz >= 32 && bsz < 128)
                       ? 128
                       : (bsz % 128 == 0)
                             ? bsz
                             : ((128 - (bsz % 128)) > 32 && bsz < 512)
                                   ? ((bsz % 64 == 0)
                                          ? bsz
                                          : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                : bsz + (64 - (bsz % 64)))
                                   : bsz + (128 - (bsz % 128));
        }
        auto auxilary_buf =
            (T*)Context::Instance().GetWorkSpace() +
            8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);
        if (q_bits == 8 or q_bits == 4) {
            launch_me((int8_t*)auxilary_buf,
                      (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                      (__half*)workspace,
                      input.size(2),
                      bsz,
                      Context::Instance().GetCurrentStream());
            run_gemm(auxilary_buf,
                     weight.data_ptr(),
                     output,
                     (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                     q_scale1.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale1.size(0),
                     Context::Instance().GetCurrentStream());
            // TODO: Reza add support for act_func_type of ReLU here
            launch_bias_gelu_int8((int8_t*)auxilary_buf,
                                  (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                                  (__half*)output,
                                  (__half*)bias.data_ptr(),
                                  out_size,
                                  bsz,
                                  Context::Instance().GetCurrentStream());
            run_gemm(auxilary_buf,
                     weight1.data_ptr(),
                     output2,
                     (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                     q_scale.data_ptr(),
                     bsz1,
                     weight1.size(0),
                     out_size,
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            run_quantize_int4((int8_t*)auxilary_buf,
                              (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                              (__half*)workspace,
                              input.size(2),
                              bsz,
                              Context::Instance().GetCurrentStream());
            run_gemm_int4(auxilary_buf,
                          weight.data_ptr(),
                          output,
                          (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                          q_scale1.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale1.size(0),
                          Context::Instance().GetCurrentStream());
            // TODO: Reza add support for act_func_type of ReLU here
            launch_bias_gelu_int4((int8_t*)auxilary_buf,
                                  (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                                  (__half*)output,
                                  (__half*)bias.data_ptr(),
                                  out_size,
                                  bsz,
                                  Context::Instance().GetCurrentStream());
            run_gemm_int4(auxilary_buf,
                          weight1.data_ptr(),
                          output2,
                          (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                          q_scale.data_ptr(),
                          bsz1,
                          weight1.size(0),
                          out_size,
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        if (bsz > 1) {
            float alpha = (T)1.0;
            float gemm_beta = (T)0.0;
            cublasSetStream(Context::Instance().GetCublasHandle(),
                            Context::Instance().GetCurrentStream());
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight.size(1),
                           bsz,
                           input.size(2),
                           &alpha,
                           &gemm_beta,
                           (T*)weight.data_ptr(),
                           workspace,
                           (T*)output,
#ifdef __HIP_PLATFORM_HCC__
                           rocblas_gemm_algo_standard);
#else
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
            if (act_func_type == ActivationFuncType::GELU) {
                launch_bias_gelu((T*)output,
                                 (T*)bias.data_ptr(),
                                 weight.size(1),
                                 bsz,
                                 Context::Instance().GetCurrentStream());
            } else if (act_func_type == ActivationFuncType::ReLU) {
                launch_bias_relu((T*)output,
                                 (T*)bias.data_ptr(),
                                 weight.size(1),
                                 bsz,
                                 Context::Instance().GetCurrentStream());
            }
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight1.size(1),
                           bsz,
                           weight1.size(0),
                           &alpha,
                           &gemm_beta,
                           (T*)weight1.data_ptr(),
                           output,
                           (T*)output2,
#ifdef __HIP_PLATFORM_HCC__
                           rocblas_gemm_algo_standard);
#else
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
        } else {
            // TODO: Reza check for activation function, need to support ReLU
            launch_input_tiled_gemm_kernel((T*)output,
                                           (T*)workspace,
                                           (T*)weight.data_ptr(),
                                           (T*)bias.data_ptr(),
                                           input.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream(),
                                           true);
            launch_input_tiled_gemm_kernel((T*)output2,
                                           (T*)output,
                                           (T*)weight1.data_ptr(),
                                           (T*)nullptr,
                                           weight1.size(0),
                                           bsz,
                                           weight1.size(1),
                                           Context::Instance().GetCurrentStream(),
                                           false);
        }
    }
    return at::from_blob(output2, {input.size(0), input.size(1), weight1.size(1)}, input.options());
}

template <typename T>
at::Tensor ds_mlp_gemm(at::Tensor& input,
                       at::Tensor& residual,
                       at::Tensor& input_bias,
                       at::Tensor& weight,
                       at::Tensor& weight1,
                       at::Tensor& bias,
                       at::Tensor& gamma,
                       at::Tensor& beta,
                       const float epsilon,
                       bool preLayerNorm,
                       bool mlp_after_attn,
                       at::Tensor& q_scale,
                       at::Tensor& q_scale1,
                       bool q_int,
                       unsigned q_bits,
                       int activation_type)
{
    T* workspace = (T*)Context::Instance().GetWorkSpace();
    int bsz = input.size(0) * input.size(1);

    auto act_func_type = static_cast<ActivationFuncType>(activation_type);

    return mlp_unfused_cublas<T>(
        workspace,
        workspace + 4 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2),
        mlp_after_attn ? input : residual,
        residual,
        input_bias,
        weight,
        weight1,
        bias,
        gamma,
        beta,
        epsilon,
        preLayerNorm,
        mlp_after_attn,
        q_scale,
        q_scale1,
        q_int,
        q_bits,
        act_func_type);
}

template <typename T>
at::Tensor fused_gemm_gelu(at::Tensor& input,
                           at::Tensor& weight,
                           at::Tensor& bias,
                           at::Tensor& weight_out,
                           const float epsilon,
                           bool preLayerNorm,
                           bool async_op,
                           at::Tensor& q_scale,
                           at::Tensor& q_scale1,
                           bool q_int,
                           unsigned q_bits)
{
    auto options = at::TensorOptions()
                       .dtype(input.options().dtype())
                       .layout(at::kStrided)
                       .device(at::kCUDA)
                       .requires_grad(false);

    T* workspace = (T*)Context::Instance().GetWorkSpace();

    auto output = torch::from_blob(
        workspace + 4 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2),
        {input.size(0), input.size(1), weight_out.size(0)},
        options);
    int bsz = input.size(0) * input.size(1);
    if (q_int) {
        int out_size = weight.size(0);
        int bsz1 = bsz;
        if (q_bits == 4 && 0) {
            // 128-aligned
            bsz1 = bsz % 128 == 0 ? bsz : (bsz / 128 + 1) * 128;
        } else {
            bsz1 = (bsz >= 32 && bsz < 128)
                       ? 128
                       : (bsz % 128 == 0)
                             ? bsz
                             : ((128 - (bsz % 128)) > 32 && bsz < 512)
                                   ? ((bsz % 64 == 0)
                                          ? bsz
                                          : ((64 - (bsz % 64)) > 32 && bsz < 32)
                                                ? ((bsz % 32 == 0) ? bsz : bsz + (32 - (bsz % 32)))
                                                : bsz + (64 - (bsz % 64)))
                                   : bsz + (128 - (bsz % 128));
        }
        auto auxilary_buf =
            workspace + 8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);
        if (q_bits == 8 or q_bits == 4) {
            launch_me((int8_t*)auxilary_buf,
                      (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                      (__half*)input.data_ptr(),
                      input.size(2),
                      bsz,
                      Context::Instance().GetCurrentStream());

            run_gemm(auxilary_buf,
                     weight.data_ptr(),
                     workspace,
                     (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                     q_scale1.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale1.size(0),
                     Context::Instance().GetCurrentStream());
            launch_bias_gelu_int8((int8_t*)auxilary_buf,
                                  (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                                  (__half*)workspace,
                                  (__half*)bias.data_ptr(),
                                  out_size,
                                  bsz,
                                  Context::Instance().GetCurrentStream());
            run_gemm(auxilary_buf,
                     weight_out.data_ptr(),
                     (T*)output.data_ptr(),
                     (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                     q_scale.data_ptr(),
                     bsz1,
                     weight_out.size(0),
                     out_size,
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            run_quantize_int4((int8_t*)auxilary_buf,
                              (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                              (__half*)input.data_ptr(),
                              input.size(2),
                              bsz,
                              Context::Instance().GetCurrentStream());

            run_gemm_int4(auxilary_buf,
                          weight.data_ptr(),
                          workspace,
                          (float*)((int8_t*)auxilary_buf + bsz1 * input.size(2)),
                          q_scale1.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale1.size(0),
                          Context::Instance().GetCurrentStream());
            launch_bias_gelu_int4((int8_t*)auxilary_buf,
                                  (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                                  (__half*)workspace,
                                  (__half*)bias.data_ptr(),
                                  out_size,
                                  bsz,
                                  Context::Instance().GetCurrentStream());
            run_gemm_int4(auxilary_buf,
                          weight_out.data_ptr(),
                          (T*)output.data_ptr(),
                          (float*)((int8_t*)auxilary_buf + bsz1 * out_size),
                          q_scale.data_ptr(),
                          bsz1,
                          weight_out.size(0),
                          out_size,
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        if (bsz > 1) {
            float alpha = (T)1.0;
            float gemm_beta = (T)0.0;
            cublasSetStream(Context::Instance().GetCublasHandle(),
                            Context::Instance().GetCurrentStream());
            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight.size(1),
                           bsz,
                           input.size(2),
                           &alpha,
                           &gemm_beta,
                           (T*)weight.data_ptr(),
                           (T*)input.data_ptr(),
                           workspace,
#ifdef __HIP_PLATFORM_HCC__
                           rocblas_gemm_algo_standard);
#else
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
            launch_bias_gelu(workspace,
                             (T*)bias.data_ptr(),
                             weight.size(1),
                             bsz,
                             Context::Instance().GetCurrentStream());

            cublas_gemm_ex(Context::Instance().GetCublasHandle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           weight_out.size(1),
                           bsz,
                           weight_out.size(0),
                           &alpha,
                           &gemm_beta,
                           (T*)weight_out.data_ptr(),
                           (T*)workspace,
                           (T*)output.data_ptr(),
#ifdef __HIP_PLATFORM_HCC__
                           rocblas_gemm_algo_standard);
#else
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#endif
        } else {
            launch_input_tiled_gemm_kernel((T*)workspace,
                                           (T*)input.data_ptr(),
                                           (T*)weight.data_ptr(),
                                           (T*)bias.data_ptr(),
                                           input.size(2),
                                           bsz,
                                           weight.size(1),
                                           Context::Instance().GetCurrentStream(),
                                           true);
            launch_input_tiled_gemm_kernel((T*)output.data_ptr(),
                                           (T*)workspace,
                                           (T*)weight_out.data_ptr(),
                                           (T*)nullptr,
                                           weight_out.size(0),
                                           bsz,
                                           weight_out.size(1),
                                           Context::Instance().GetCurrentStream(),
                                           false);
        }
    }
    return output;
}

template <typename T>
at::Tensor& residual_add_bias(at::Tensor& hidden_state,
                              at::Tensor& residual,
                              const at::Tensor& attention_output,
                              const at::Tensor& attention_bias,
                              const at::Tensor& final_bias,
                              const int mp_size,
                              const bool mlp_after_attn,
                              const bool add_bias,
                              const bool preln)
{
    int bsz = residual.size(0) * residual.size(1);
    int hidden_size = residual.size(2);
    if (mlp_after_attn)
        launch_bias_residual(static_cast<T*>(residual.data_ptr()),
                             static_cast<T*>(hidden_state.data_ptr()),
                             static_cast<T*>(attention_output.data_ptr()),
                             static_cast<T*>(final_bias.data_ptr()),
                             static_cast<T*>(attention_bias.data_ptr()),
                             bsz,
                             hidden_size,
                             mp_size,
                             preln,
                             Context::Instance().GetCurrentStream());
    else
        launch_gptj_residual_add<T>(
            static_cast<T*>(residual.data_ptr()),
            static_cast<T*>(hidden_state.data_ptr()),
            static_cast<T*>(attention_output.data_ptr()),
            static_cast<T*>(final_bias.data_ptr()),
            static_cast<T*>((add_bias ? attention_bias.data_ptr() : nullptr)),
            hidden_size,
            bsz,
            mp_size,
            Context::Instance().GetCurrentStream());
    return residual;
}

std::vector<at::Tensor> apply_rotary_pos_emb(at::Tensor& mixed_query,
                                             at::Tensor& key_layer,
                                             unsigned rotary_dim,
                                             unsigned offset,
                                             unsigned num_heads,
                                             bool rotate_half,
                                             bool rotate_every_two)
{
    auto query_cont = mixed_query.contiguous();
    auto key_cont = key_layer.contiguous();

    unsigned bsz = mixed_query.size(0);
    unsigned head_size = mixed_query.size(2) / num_heads;
    unsigned seq_len = mixed_query.size(1);

    if (mixed_query.scalar_type() == at::kFloat)
        launch_apply_rotary_pos_emb<float>((float*)query_cont.data_ptr(),
                                           (float*)key_cont.data_ptr(),
                                           head_size,
                                           seq_len,
                                           rotary_dim,
                                           offset,
                                           num_heads,
                                           bsz,
                                           rotate_half,
                                           rotate_every_two,
                                           Context::Instance().GetCurrentStream(),
                                           Context::Instance().GetMaxTokenLenght());
    else
        launch_apply_rotary_pos_emb<__half>((__half*)query_cont.data_ptr(),
                                            (__half*)key_cont.data_ptr(),
                                            head_size,
                                            seq_len,
                                            rotary_dim,
                                            offset,
                                            num_heads,
                                            bsz,
                                            rotate_half,
                                            rotate_every_two,
                                            Context::Instance().GetCurrentStream(),
                                            Context::Instance().GetMaxTokenLenght());
    return {query_cont, key_cont};
}

at::Tensor moe_res_matmul(at::Tensor& moe_res, at::Tensor& coef, at::Tensor& output)
{
    int M = moe_res.size(0) * moe_res.size(1);
    int N = moe_res.size(2);
    // Context::Instance().SynchComm();
    if (moe_res.scalar_type() == at::kFloat) {
        launch_moe_res_matmul<float>((float*)moe_res.data_ptr(),
                                     (float*)coef.data_ptr(),
                                     (float*)output.data_ptr(),
                                     M,
                                     N,
                                     at::cuda::getCurrentCUDAStream());
    } else {
        launch_moe_res_matmul<__half>((__half*)moe_res.data_ptr(),
                                      (__half*)coef.data_ptr(),
                                      (__half*)output.data_ptr(),
                                      M,
                                      N,
                                      at::cuda::getCurrentCUDAStream());
    }
    return output;
}

template <typename T>
void TransformerEncoder(at::Tensor& input,
                        at::Tensor& input_mask,
                        std::vector<at::Tensor>& input_norm,
                        std::vector<at::Tensor>& attn_weights,
                        std::vector<at::Tensor>& attn_biases,
                        std::vector<at::Tensor>& attn_norm,
                        std::vector<at::Tensor>& mlp_weights,
                        std::vector<at::Tensor>& mlp_biases,
                        int num_heads,
                        bool preln,
                        float epsilon,
                        float norm_factor,
                        bool q_int,
                        unsigned q_bits,
                        at::Tensor& q_scale,
                        at::Tensor& q_scale1,
                        at::Tensor& q_scale2,
                        bool enable_qkv_quantization,
                        at::Tensor& q_scale3)
{
    unsigned bsz = input.size(0);
    unsigned hidden_dim = input.size(2);
    unsigned head_size = hidden_dim / num_heads;
    unsigned _seq_length = input.size(1);
    unsigned seq2 = _seq_length * _seq_length;
    unsigned seq_head = _seq_length * head_size;
    cudaStream_t new_stream = at::cuda::getCurrentCUDAStream();
    auto cub_handle = Context::Instance().GetCublasHandle();

    T* workspace = (T*)(Context::Instance().GetWorkSpace());
    if (!workspace) {
        allocate_workspace<T>(hidden_dim, bsz, 1);
        workspace = (T*)Context::Instance().GetWorkSpace();
    }
    cublasSetStream(cub_handle, new_stream);
    size_t small_buf_size = bsz * _seq_length * hidden_dim;
    T* buf_0 = workspace;                   // 1
    T* buf_1 = buf_0 + small_buf_size;      // 3
    T* buf_2 = buf_1 + 3 * small_buf_size;  // 3
    T* buf_3 = buf_2 + 3 * small_buf_size;  // 1
    T* buf_4 = buf_3 + small_buf_size;      // 1
    T* buf_5 = buf_4 + small_buf_size;      // 1

    int bsz_seq = bsz * _seq_length;
    int bsz1;
    if (q_bits == 4) {
        // 128-aligned
        bsz1 = bsz_seq % 128 == 0 ? bsz_seq : (bsz_seq / 128 + 1) * 128;
    } else {
        bsz1 =
            (bsz_seq >= 32 && bsz_seq < 128)
                ? 128
                : (bsz_seq % 128 == 0)
                      ? bsz_seq
                      : ((128 - (bsz_seq % 128)) > 32 && bsz_seq < 512)
                            ? ((bsz_seq % 64 == 0)
                                   ? bsz_seq
                                   : ((64 - (bsz_seq % 64)) > 32 && bsz_seq < 32)
                                         ? ((bsz_seq % 32 == 0) ? bsz_seq
                                                                : bsz_seq + (32 - (bsz_seq % 32)))
                                         : bsz_seq + (64 - (bsz_seq % 64)))
                            : bsz_seq + (128 - (bsz_seq % 128));
    }
    auto aux_buff = (T*)Context::Instance().GetWorkSpace() +
                    8 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);

    auto aux_buff1 = (T*)Context::Instance().GetWorkSpace() +
                     12 * input.size(0) * Context::Instance().GetMaxTokenLenght() * input.size(2);

    T* input_ptr = (T*)input.data_ptr();
    float alpha = (T)1.0;
    float gemm_beta = (T)0.0;
    int bsz_heads = bsz * num_heads;
    if (preln)
        launch_layer_norm(buf_0,
                          input_ptr,
                          (T*)input_norm[0].data_ptr(),
                          (T*)input_norm[1].data_ptr(),
                          epsilon,
                          bsz_seq,
                          hidden_dim,
                          new_stream);
    if (q_int and enable_qkv_quantization) {
        int out_size = attn_weights[0].size(0);
        if (q_bits == 8) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)(preln ? buf_0 : input_ptr),
                      input.size(2),
                      bsz_seq,
                      Context::Instance().GetCurrentStream());
            run_gemm(aux_buff,
                     attn_weights[0].data_ptr(),
                     buf_1,
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale3.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale3.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            assert(q_bits == 4);
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)(preln ? buf_0 : input_ptr),
                              input.size(2),
                              bsz_seq,
                              Context::Instance().GetCurrentStream());
            run_gemm_int4(aux_buff,
                          attn_weights[0].data_ptr(),
                          buf_1,
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale3.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale3.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        cublas_gemm_ex(cub_handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       attn_weights[0].size(1),
                       bsz_seq,
                       hidden_dim,
                       &alpha,
                       &gemm_beta,
                       (T*)attn_weights[0].data_ptr(),
                       preln ? buf_0 : input_ptr,
                       buf_1,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    if (_seq_length >= 32 || (bsz * (hidden_dim / num_heads)) > 128) {
        launch_bias_add_transform_0213<T>(buf_2,
                                          buf_1,
                                          (T*)attn_biases[0].data_ptr(),
                                          bsz,
                                          _seq_length,
                                          hidden_dim,
                                          num_heads,
                                          new_stream,
                                          3);
        alpha = norm_factor;
        cublas_strided_batched_gemm(cub_handle,
                                    _seq_length,
                                    _seq_length,
                                    head_size,
                                    &alpha,
                                    &gemm_beta,
                                    buf_2 + small_buf_size,
                                    buf_2,
                                    buf_3,
                                    CUBLAS_OP_T,
                                    CUBLAS_OP_N,
                                    seq_head,
                                    seq_head,
                                    seq2,
                                    bsz_heads,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        alpha = 1.0;
        int mask_stride = 1;
        if (input_mask.sizes().size() > 2) mask_stride = input_mask.size(2);
        launch_attn_softmax_v2(buf_3,
                               (T*)input_mask.data_ptr(),
                               (T*)nullptr,
                               1.0,
                               false,
                               true,
                               false,
                               1,
                               bsz,
                               num_heads,
                               _seq_length,
                               _seq_length,
                               0,
                               mask_stride,
                               1.0,
                               new_stream);

        cublas_strided_batched_gemm(cub_handle,
                                    head_size,
                                    _seq_length,
                                    _seq_length,
                                    &alpha,
                                    &gemm_beta,
                                    buf_2 + 2 * small_buf_size,
                                    buf_3,
                                    buf_0,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    seq_head,
                                    seq2,
                                    seq_head,
                                    bsz_heads,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        launch_attn_softmax_context((T*)buf_0,
                                    (T*)buf_1,
                                    (T*)input_mask.data_ptr(),
                                    norm_factor,
                                    (T*)nullptr,
                                    (T*)nullptr,
                                    (T*)attn_biases[0].data_ptr(),
                                    true,
                                    false,
                                    true,
                                    bsz,
                                    num_heads,
                                    hidden_dim / num_heads,
                                    _seq_length,
                                    _seq_length,
                                    _seq_length,
                                    1.0,
                                    at::cuda::getCurrentCUDAStream());
    }
    launch_transform4d_0213<T>(
        buf_2, buf_0, bsz, num_heads, _seq_length, hidden_dim, new_stream, 1);
    if (q_int) {
        int out_size = attn_weights[1].size(0);
        if (q_bits == 8) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)buf_2,
                      input.size(2),
                      bsz_seq,
                      Context::Instance().GetCurrentStream());
            run_gemm(aux_buff,
                     attn_weights[1].data_ptr(),
                     buf_1,
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale2.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale2.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            assert(q_bits == 4);
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)buf_2,
                              input.size(2),
                              bsz_seq,
                              Context::Instance().GetCurrentStream());
            run_gemm_int4(aux_buff,
                          attn_weights[1].data_ptr(),
                          buf_1,
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale2.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale2.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        cublas_gemm_ex(cub_handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       attn_weights[1].size(1),
                       bsz_seq,
                       hidden_dim,
                       &alpha,
                       &gemm_beta,
                       (T*)attn_weights[1].data_ptr(),
                       buf_2,
                       buf_1,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    launch_residual_layer_norm1<T>(buf_4,
                                   buf_1,
                                   input_ptr,
                                   (T*)attn_biases[1].data_ptr(),
                                   (T*)attn_norm[0].data_ptr(),
                                   (T*)attn_norm[1].data_ptr(),
                                   epsilon,
                                   bsz_seq,
                                   hidden_dim,
                                   new_stream);
    if (q_int) {
        int out_size = mlp_weights[0].size(0);
        if (q_bits == 8) {
            launch_me((int8_t*)aux_buff,
                      (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                      (__half*)buf_4,
                      input.size(2),
                      bsz_seq,
                      Context::Instance().GetCurrentStream());

            run_gemm(aux_buff,
                     mlp_weights[0].data_ptr(),
                     aux_buff1,
                     (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                     q_scale1.data_ptr(),
                     bsz1,
                     out_size,
                     input.size(2),
                     bsz1,
                     q_scale1.size(0),
                     Context::Instance().GetCurrentStream());
            launch_bias_gelu_int8((int8_t*)aux_buff,
                                  (float*)((int8_t*)aux_buff + bsz1 * out_size),
                                  (__half*)aux_buff1,
                                  (__half*)mlp_biases[0].data_ptr(),
                                  out_size,
                                  bsz_seq,
                                  Context::Instance().GetCurrentStream());
            run_gemm(aux_buff,
                     mlp_weights[1].data_ptr(),
                     (T*)buf_5,
                     (float*)((int8_t*)aux_buff + bsz1 * out_size),
                     q_scale.data_ptr(),
                     bsz1,
                     mlp_weights[1].size(0),
                     out_size,
                     bsz1,
                     q_scale.size(0),
                     Context::Instance().GetCurrentStream());
        } else {
            assert(q_bits == 4);
            // std::cout << "mlp q_bits == 4" << std::endl;
            run_quantize_int4((int8_t*)aux_buff,
                              (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                              (__half*)buf_4,
                              input.size(2),
                              bsz_seq,
                              new_stream);

            run_gemm_int4(aux_buff,
                          mlp_weights[0].data_ptr(),
                          aux_buff1,
                          (float*)((int8_t*)aux_buff + bsz1 * input.size(2)),
                          q_scale1.data_ptr(),
                          bsz1,
                          out_size,
                          input.size(2),
                          bsz1,
                          q_scale1.size(0),
                          Context::Instance().GetCurrentStream());

            launch_bias_gelu_int4((int8_t*)aux_buff,
                                  (float*)((int8_t*)aux_buff + bsz1 * out_size),
                                  (__half*)aux_buff1,
                                  (__half*)mlp_biases[0].data_ptr(),
                                  out_size,
                                  bsz_seq,
                                  Context::Instance().GetCurrentStream());
            run_gemm_int4(aux_buff,
                          mlp_weights[1].data_ptr(),
                          (T*)buf_5,
                          (float*)((int8_t*)aux_buff + bsz1 * out_size),
                          q_scale.data_ptr(),
                          bsz1,
                          mlp_weights[1].size(0),
                          out_size,
                          bsz1,
                          q_scale.size(0),
                          Context::Instance().GetCurrentStream());
        }
    } else {
        cublas_gemm_ex(cub_handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       mlp_weights[0].size(1),
                       bsz_seq,
                       hidden_dim,
                       &alpha,
                       &gemm_beta,
                       (T*)mlp_weights[0].data_ptr(),
                       buf_4,
                       buf_0,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        launch_bias_gelu(
            buf_0, (T*)mlp_biases[0].data_ptr(), mlp_weights[0].size(1), bsz_seq, new_stream);

        cublas_gemm_ex(cub_handle,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       mlp_weights[1].size(1),
                       bsz_seq,
                       mlp_weights[0].size(1),
                       &alpha,
                       &gemm_beta,
                       (T*)mlp_weights[1].data_ptr(),
                       buf_0,
                       buf_5,
                       CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (!preln) {
        launch_residual_layer_norm1<T>(input_ptr,
                                       buf_5,
                                       buf_4,
                                       (T*)mlp_biases[1].data_ptr(),
                                       (T*)input_norm[0].data_ptr(),
                                       (T*)input_norm[1].data_ptr(),
                                       epsilon,
                                       bsz_seq,
                                       hidden_dim,
                                       new_stream);
    } else
        launch_bias_residual1(input_ptr,
                              buf_5,
                              buf_1,
                              (T*)mlp_biases[1].data_ptr(),
                              (T*)attn_biases[1].data_ptr(),
                              bsz_seq,
                              hidden_dim,
                              preln,
                              new_stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_fp32", &ds_softmax<float>, "DeepSpeed SoftMax with fp32 (CUDA)");
    m.def("softmax_fp16", &ds_softmax<__half>, "DeepSpeed SoftMax with fp16 (CUDA)");
    m.def(
        "softmax_context_fp32", &ds_softmax_context<float>, "DeepSpeed attention with fp32 (CUDA)");
    m.def("softmax_context_fp16",
          &ds_softmax_context<__half>,
          "DeepSpeed attention with fp16 (CUDA)");
    m.def("bias_gelu_fp32", &ds_bias_gelu<float>, "DeepSpeed Gelu with fp32 (CUDA)");
    m.def("bias_gelu_fp16", &ds_bias_gelu<__half>, "DeepSpeed Gelu with fp16 (CUDA)");
    m.def("bias_add_fp32", &ds_bias_add<float>, "DeepSpeed Bias Add with fp32 (CUDA)");
    m.def("bias_add_fp16", &ds_bias_add<__half>, "DeepSpeed Gelu with fp16 (CUDA)");
    m.def("bias_relu_fp32", &ds_bias_relu<float>, "DeepSpeed ReLU with fp32 (CUDA)");
    m.def("bias_relu_fp16", &ds_bias_relu<__half>, "DeepSpeed ReLU with fp16 (CUDA)");
    m.def("bias_residual_fp32",
          &ds_bias_residual<float>,
          "DeepSpeed residual-bias add with fp32 (CUDA)");
    m.def("bias_residual_fp16",
          &ds_bias_residual<__half>,
          "DeepSpeed residual-bias add with fp16 (CUDA)");
    m.def("layer_norm_fp32", &ds_layernorm<float>, "DeepSpeed layer-norm with fp32 (CUDA)");
    m.def("layer_norm_fp16", &ds_layernorm<__half>, "DeepSpeed layer-norm with fp16 (CUDA)");
    m.def("qkv_gemm_fp32", &ds_qkv_gemm<float>, "DeepSpeed qkv gemm with fp32 (CUDA)");
    m.def("qkv_gemm_fp16", &ds_qkv_gemm<__half>, "DeepSpeed qkv gemm with fp16 (CUDA)");
    m.def("mlp_gemm_fp32", &ds_mlp_gemm<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("mlp_gemm_fp16", &ds_mlp_gemm<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("vector_matmul_fp32", &ds_vector_matmul<float>, "DeepSpeed vector-MM with fp32 (CUDA)");
    m.def("vector_matmul_fp16", &ds_vector_matmul<__half>, "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("linear_layer_fp32", &ds_linear_layer<float>, "DeepSpeed linear_layer with fp32 (CUDA)");
    m.def("linear_layer_fp16", &ds_linear_layer<__half>, "DeepSpeed linear_layer with fp16 (CUDA)");
    m.def("fused_gemm_gelu_fp32", &fused_gemm_gelu<float>, "DeepSpeed mlp with fp32 (CUDA)");
    m.def("fused_gemm_gelu_fp16", &fused_gemm_gelu<__half>, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("residual_add_bias_fp32",
          &residual_add_bias<float>,
          "DeepSpeed residual add with fp32 (CUDA)");
    m.def("residual_add_bias_fp16",
          &residual_add_bias<__half>,
          "DeepSpeed residual add with fp16 (CUDA)");
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "DeepSpeed mlp with fp16 (CUDA)");
    m.def("einsum_sec_sm_ecm_fp32",
          &einsum_sec_sm_ecm<float>,
          "DeepSpeed vector-MM with fp32 (CUDA)");

    m.def("einsum_sec_sm_ecm_fp16",
          &einsum_sec_sm_ecm<__half>,
          "DeepSpeed vector-MM with fp16 (CUDA)");
    m.def("moe_res_matmul", &moe_res_matmul, "DeepSpeed moe residual matmul (CUDA)");
    m.def("encoder_fp32",
          &TransformerEncoder<float>,
          "DeepSpeed transformerEncoder with fp32 (CUDA)");
    m.def("encoder_fp16",
          &TransformerEncoder<__half>,
          "DeepSpeed transformerEncoder with fp16 (CUDA)");
}
