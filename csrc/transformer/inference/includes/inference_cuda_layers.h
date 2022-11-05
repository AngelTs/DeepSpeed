/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#pragma once

#include "ds_kernel_utils.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#define MAX_WARP_NUM 32
#define WARP_SIZE 32

#define MAX_THREADS 1024
#define SMs 80

#define MAX_REGISTERS 256

struct int4x2_t {
    int8_t high : 4;
    int8_t low : 4;
};

template <typename T>
void launch_attn_softmax_v2(T* vals,
                            T* mask,
                            T* alibi,
                            float layer_scale,
                            bool triangular,
                            bool recompute,
                            bool local_attention,
                            int window_size,
                            int batch_size,
                            int heads,
                            int num_seq,
                            int sequence_length,
                            int offset,
                            int mask_stride,
                            int mp_size,
                            cudaStream_t stream);

template <typename T>
void launch_attn_softmax(T*, const T*, int, int, int, cudaStream_t);
// Fused bias add with gelu activation
template <typename T>
void launch_bias_gelu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);

// Fused bias add with relu activation
template <typename T>
void launch_bias_relu(T* input,
                      const T* bias,
                      int intermediate_size,
                      int batch_size,
                      cudaStream_t stream);

template <typename T>
void launch_bias_add(T* input, const T* bias, int hidden_size, int batch_size, cudaStream_t stream);

template <typename T>
void launch_bias_residual(T* input,
                          T* output,
                          T* attn,
                          T* bias,
                          T* attn_bias,
                          int batch,
                          int hidden_dim,
                          int mp_size,
                          bool preln,
                          cudaStream_t stream);

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);

template <typename T>
void launch_residual_layer_norm(T* norm,
                                T* res_add,
                                T* vals,
                                T* residual,
                                const T* bias,
                                const T* gamma,
                                const T* beta,
                                float epsilon,
                                int batch_size,
                                int hidden_dim,
                                bool preLN,
                                bool mlp_after_attn,
                                cudaStream_t stream);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream);

template <typename T>
void launch_fused_residual_ln(T* output,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              cudaStream_t stream);

void launch_fused_residual_ln_quant(int8_t* out_int8,
                                    __half* output,
                                    float* scales,
                                    const __half* vals,
                                    const __half* residual,
                                    const __half* bias,
                                    const __half* gamma,
                                    const __half* beta,
                                    float epsilon,
                                    int rows,
                                    int elems_per_row,
                                    cudaStream_t stream);

void launch_fused_residual_ln_quant_int4(int8_t* out_int8,
                                         __half* output,
                                         float* scales,
                                         const __half* vals,
                                         const __half* residual,
                                         const __half* bias,
                                         const __half* gamma,
                                         const __half* beta,
                                         float epsilon,
                                         int rows,
                                         int elems_per_row,
                                         cudaStream_t stream);
template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       unsigned merge_count,
                       cudaStream_t stream);

template <typename T>
void launch_dequantize(T* output,
                       const int8_t* input,
                       const float* qscale,
                       unsigned output_size,
                       unsigned hidden_dim,
                       unsigned groups,
                       cudaStream_t stream);
template <typename T>
void launch_gptj_residual_add(T* input,
                              T* output,
                              T* attn,
                              T* bias,
                              T* attn_bias,
                              int batch,
                              int head_size,
                              int mp_size,
                              cudaStream_t stream);

template <typename T>
void launch_apply_rotary_pos_emb(T* mixed_query,
                                 T* key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 bool rotate_half,
                                 bool rotate_every_two,
                                 cudaStream_t stream,
                                 int max_out_tokens);

template <typename T>
void launch_moe_res_matmul(T* residual,
                           T* coef,
                           T* mlp_out,
                           int seq_len,
                           int hidden_dim,
                           cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    cudaStream_t stream,
                                    bool add_gelu = false);
template <typename T>
void launch_input_tiled_gemm_kernel_v2(T* output,
                                       const T* vals,
                                       const T* weight,
                                       const T* bias,
                                       T* block_sums,
                                       unsigned int hidden_dim,
                                       unsigned int input_size,
                                       unsigned int output_size,
                                       bool add_gelu,
                                       cudaStream_t stream);

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    T* intermediate,
                                    const T* vals,
                                    const T* weight,
                                    const T* gamma,
                                    const T* beta,
                                    float epsilon,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    int intm_size,
                                    cudaStream_t stream);

// 4D transform [0, 1, 2, 3] -> [0, 2, 1, 3]
template <typename T>
void launch_transform4d_0213(T* out,
                             const T* in,
                             int batch_size,
                             int heads,
                             int seq_length,
                             int hidden_dim,
                             cudaStream_t stream,
                             int trans_count);

template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    const T* vals,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    int hidden_dim,
                                    int heads,
                                    cudaStream_t stream,
                                    int trans_count);
// Custom bias add
template <typename T>
void launch_bias_add_transform_0213(T* outputs,
                                    T* vals,
                                    T* vals1,
                                    const T* vals2,
                                    const T* bias,
                                    int batch_size,
                                    int seq_length,
                                    unsigned seq_offset,
                                    int seq_length1,
                                    int hidden_dim,
                                    int heads,
                                    int rotary_dim,
                                    bool rotate_half,
                                    bool rotate_every_two,
                                    cudaStream_t stream,
                                    int trans_count,
                                    int max_out_tokens);
template <typename T>
void launch_transform_scale(T* vals,
                            T* query,
                            T* kv_cache,
                            int batch_size,
                            int seq_length,
                            unsigned cur_tokens,
                            size_t value_offset,
                            unsigned hidden_dim,
                            int heads,
                            cudaStream_t stream,
                            int trans_count,
                            float norm_factor,
                            size_t max_token_length);

void run_gemm(void* A,
              void* B,
              void* C,
              void* a,
              void* aa,
              int M,
              int N,
              int K,
              int groups,
              int groups1,
              cudaStream_t stream);

void run_gemm_int4(void* A,
                   void* B,
                   void* C,
                   void* a,
                   void* aa,
                   int M,
                   int N,
                   int K,
                   int groups,
                   int groups1,
                   cudaStream_t stream);

void run_quantize_int4(int8_t* output,
                       float* scales,
                       __half* input,
                       int intermediate_size,
                       int batch_size,
                       cudaStream_t stream);

void launch_me(int8_t* output,
               float* scales,
               __half* input,
               int intermediate_size,
               int batch_size,
               cudaStream_t stream);

void launch_bias_gelu_int4(int8_t* output,
                           float* scales,
                           __half* input,
                           const __half* bias,
                           int intermediate_size,
                           int batch_size,
                           cudaStream_t stream);

void launch_bias_gelu_int8(int8_t* output,
                           float* scales,
                           __half* input,
                           const __half* bias,
                           int intermediate_size,
                           int batch_size,
                           cudaStream_t stream);
template <typename T>
void launch_residual_layer_norm_int8(int8_t* res_add,
                                     float* scales,
                                     T* vals,
                                     T* residual,
                                     const T* bias,
                                     const T* gamma,
                                     const T* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     bool preLN,
                                     int mp_size,
                                     cudaStream_t stream);
template <typename T>
void launch_layer_norm_int8(int8_t* out,
                            float* scales,
                            T* vals,
                            const T* gamma,
                            const T* beta,
                            float epsilon,
                            int batch_size,
                            int hidden_dim,
                            cudaStream_t stream);

template <typename T>
void launch_attn_softmax_context(T* out,
                                 T* query,
                                 T* mask,
                                 float norm_factor,
                                 T* key_merged,
                                 T* merged_value,
                                 T* attn_bias,
                                 bool merging,
                                 bool triangular,
                                 bool recompute,
                                 int batch_size,
                                 int heads,
                                 int head_size,
                                 int value_length,
                                 int num_seq,
                                 int sequence_length,
                                 float scale,
                                 cudaStream_t stream);

template <typename T>
void launch_residual_layer_norm1(T* norm,
                                 T* vals,
                                 T* residual,
                                 const T* bias,
                                 const T* gamma,
                                 const T* beta,
                                 float epsilon,
                                 int batch_size,
                                 int hidden_dim,
                                 cudaStream_t stream);
template <typename T>
void launch_bias_residual1(T* input,
                           const T* residual,
                           const T* output,
                           const T* bias,
                           const T* bias1,
                           int batch,
                           int intermediate_size,
                           bool preln,
                           cudaStream_t stream);

void launch_act_quant(int8_t* output_data,
                      float* scales,
                      const __half* input_data,
                      int groups,
                      int elems_per_group,
                      cudaStream_t stream);

void launch_act_quant_int4(int8_t* output_data,
                           float* scales,
                           const __half* input_data,
                           int groups,
                           int elems_per_group,
                           cudaStream_t stream);

void launch_gelu_quant(int8_t* output_data,
                       float* scales,
                       const __half* input_data,
                       const __half* bias_data,
                       int groups,
                       int elems_per_group,
                       cudaStream_t stream);

void launch_gelu_quant_int4(int8_t* output_data,
                            float* scales,
                            const __half* input_data,
                            const __half* bias_data,
                            int groups,
                            int elems_per_group,
                            cudaStream_t stream);

void launch_gelu_quant(int8_t* output_data,
                       float* scales,
                       const __half* input_data,
                       const __half* gamma,
                       const __half* beta,
                       float epsilon,
                       int groups,
                       int elems_per_group,
                       cudaStream_t stream);

void launch_dequant(__half* output,
                    const int8_t* quantized_data,
                    const float* scales,
                    int elems_per_group,
                    int total_elems,
                    cudaStream_t stream);

void launch_dequant_int4(__half* output,
                         const int8_t* quantized_data,
                         const float* scales,
                         int elems_per_group,
                         int total_elems,
                         cudaStream_t stream);

void launch_dequant_reduce(int8_t* reduced_data,
                           float* reduced_scales,
                           const int8_t* input_data,
                           const float* input_scales,
                           int num_gpus,
                           int num_bits,
                           int out_groups,
                           int elems_per_out_group,
                           int elems_per_in_tensor,
                           int groups_per_in_tensor,
                           int elems_per_in_group,
                           cudaStream_t stream);
