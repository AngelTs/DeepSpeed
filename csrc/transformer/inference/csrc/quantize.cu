#include <cstdio>
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace act_quant {
constexpr int granularity = 16;
constexpr int h_per_load = granularity / sizeof(__half);
constexpr int h2_per_load = granularity / sizeof(__half2);

constexpr int threads = 256;
// BRITTLE
constexpr int warp_size = 32;
constexpr int num_warps = threads / warp_size;

constexpr int internal_unroll = 2;
constexpr int h_per_step = h_per_load * internal_unroll;

// Currently hardcoded, can re-evaluate in the future
constexpr int q_bits = 8;
constexpr int q_range = 1 << q_bits;
}  // namespace act_quant

/*
Sum reduce helper
*/
__device__ __forceinline__ float reduce_sum(cg::thread_block& tb,
                                            cg::thread_block_tile<act_quant::warp_size>& warp,
                                            float* sum_buffer,
                                            float partial_sum_arg)
{
    float partial_sum = partial_sum_arg;

#pragma unroll
    for (int i = act_quant::warp_size / 2; i > 0; i /= 2) {
        partial_sum += warp.shfl_down(partial_sum, i);
    }

    // If we have more than one warp, then we need another stage of reduction.
    if (warp.meta_group_size() > 1) {
        if (warp.thread_rank() == 0) sum_buffer[warp.meta_group_rank()] = partial_sum;

        // Safe in the conditional since all threads will evaluate the if-statement identically
        tb.sync();

        if (warp.meta_group_rank() == 0) {
            float r_sum = 0.f;
            if (warp.thread_rank() < warp.meta_group_size()) r_sum = sum_buffer[warp.thread_rank()];

#pragma unroll
            for (int i = act_quant::num_warps / 2; i > 0; i /= 2) {
                r_sum += warp.shfl_down(r_sum, i);
            }

            if (warp.thread_rank() == 0) { sum_buffer[0] = r_sum; }
        }

        // Safe in the conditional since all threads will evaluate the if-statement identically
        tb.sync();

        return sum_buffer[0];
    } else {
        // Otherwise broadcast from thread 0 and continue
        return warp.shfl(partial_sum, 0);
    }
}

/*
Quantization reduction helper. Input is the max value seen by each thread,
returns the quantization scale. Inverse still needs to be stored to global
memory by the caller.
*/
__device__ __forceinline__ float get_scale(cg::thread_block& tb,
                                           cg::thread_block_tile<act_quant::warp_size>& warp,
                                           float* max_buffer,
                                           float thread_max_arg)
{
    float thread_max_f = thread_max_arg;

#pragma unroll
    for (int i = act_quant::warp_size / 2; i > 0; i /= 2) {
        thread_max_f = fmaxf(thread_max_f, warp.shfl_down(thread_max_f, i));
    }

    // If we have more than one warp, then we need another stage of reduction.
    if (warp.meta_group_size() > 1) {
        if (warp.thread_rank() == 0) max_buffer[warp.meta_group_rank()] = thread_max_f;

        // Safe in the conditional since all threads will evaluate the if-statement identically
        tb.sync();

        if (warp.meta_group_rank() == 0) {
            float r_max = 0.f;
            if (warp.thread_rank() < warp.meta_group_size()) r_max = max_buffer[warp.thread_rank()];

#pragma unroll
            for (int i = act_quant::num_warps / 2; i > 0; i /= 2) {
                r_max = max(r_max, warp.shfl_down(r_max, i));
            }

            const float quantization_scale = act_quant::q_range / (2 * r_max);

            if (warp.thread_rank() == 0) { max_buffer[0] = quantization_scale; }
        }

        // Safe in the conditional since all threads will evaluate the if-statement identically
        tb.sync();

        return max_buffer[0];
    } else {
        // Otherwise broadcast from thread 0 and continue
        const float quantization_scale = act_quant::q_range / (2 * thread_max_f);

        return warp.shfl(quantization_scale, 0);
        // return warp.shfl(thread_max_f, 0);
    }
}

/*
Quantization inner loop helper.
*/
template <int q_bits>
__device__ __forceinline__ void quant_16_bytes(int8_t* local_output,
                                               const __half* data,
                                               float scale)
{
    constexpr int32_t q_min = -(1 << (q_bits - 1));
    constexpr int32_t q_max = (1 << (q_bits - 1)) - 1;
    constexpr int32_t elems = 16 / sizeof(__half);

#pragma unroll
    for (int i = 0; i < elems; i++) {
        // TODO(cmikeh2): refactor to use conversion utils
        float data_f = __half2float(data[i]) * scale;
        int32_t data_i32 = __float2int_rn(data_f);
        data_i32 = min(max(data_i32, q_min), q_max);
        local_output[i] = (int8_t)data_i32;
    }
}

/*
Input could be in __half2, just convert and pass to the base implementation.
*/
template <int q_bits>
__device__ __forceinline__ void quant_16_bytes(int8_t* local_output,
                                               const __half2* data,
                                               float scale)
{
    const __half* data_cast = reinterpret_cast<const __half*>(data);
    quant_16_bytes<q_bits>(local_output, data_cast, scale);
}

DS_D_INLINE
__half2 h2_max(__half2 lhs, __half2 rhs)
{
#if __CUDA_ARCH__ >= 800
    return __hmax2(lhs, rhs);
#else
    __half2 ret_val;
    ret_val.x = (lhs.x > rhs.x) ? lhs.x : rhs.x;
    ret_val.y = (lhs.y > rhs.y) ? lhs.y : rhs.y;
    return ret_val;
#endif
}

template <int UNROLL>
__device__ void device_quantize(__half2* local_buffer,
                                float* __restrict__ scales,
                                int8_t* __restrict__ output_data,
                                const int& base_offset,
                                const int& elem_offset,
                                const int& stride,
                                const int& elems_per_group)
{
    // Conservative allocation, shared memory won't be an occupancy limiter though
    __shared__ float max_buffer[act_quant::num_warps];

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<act_quant::warp_size> warp =
        cg::tiled_partition<act_quant::warp_size>(tb);

    float2 zeros = {0.f, 0.f};
    __half2 thread_max_h2 = __float22half2_rn(zeros);
#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize
        __half2* iteration_buffer =
            local_buffer + i * act_quant::internal_unroll * act_quant::h2_per_load;
        // TODO(cmikeh2): this might be faster with a tree reduce
        // but in general should mem bottlenecked so not a priority
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll * act_quant::h2_per_load; j++) {
            thread_max_h2 = h2_max(thread_max_h2, __habs2(iteration_buffer[j]));
        }
    }
    float2 thread_max_f2 = __half22float2(thread_max_h2);
    float thread_max_f = fmaxf(thread_max_f2.x, thread_max_f2.y);

    float q_scale = get_scale(tb, warp, max_buffer, thread_max_f);
    if (tb.thread_index().x == 0) scales[tb.group_index().x] = 1 / q_scale;

    int8_t* output_base = output_data + base_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * act_quant::internal_unroll; i++) {
        int8_t local_output[act_quant::h_per_load];

        quant_16_bytes<act_quant::q_bits>(
            local_output, local_buffer + i * act_quant::h2_per_load, q_scale);

        if (elem_offset + i * stride < elems_per_group) {
            mem_access::store_global<act_quant::granularity / 2>(output_base + i * stride,
                                                                 local_output);
        }
    }
}

/*
Pure quantization kernel with no fusion.
*/
template <int UNROLL>
__global__ void activation_quantization(int8_t* __restrict__ output_data,
                                        float* __restrict__ scales,
                                        const __half* __restrict__ input_data,
                                        int groups,
                                        int elems_per_group)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<act_quant::warp_size> warp =
        cg::tiled_partition<act_quant::warp_size>(tb);

    // Indexing offsets
    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * act_quant::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * act_quant::h_per_load;

    const __half* input_base = input_data + base_offset;

    __half2 local_buffer[UNROLL * act_quant::internal_unroll * act_quant::h2_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize
        __half2* iteration_buffer =
            local_buffer + i * act_quant::internal_unroll * act_quant::h2_per_load;
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll; j++) {
            const int iteration = i * act_quant::internal_unroll + j;
            mem_access::load_global<act_quant::granularity>(
                iteration_buffer + j * act_quant::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }
    }

    device_quantize<UNROLL>(
        local_buffer, scales, output_data, base_offset, elem_offset, stride, elems_per_group);
}

// Copied from gelu.cu, probably should just be in a header file
inline __device__ float q_gelu(const float x)
{
    const float sqrt_param = 0.79788456080286535587989211986876f;
    const float mul_param = 0.044715;
    return x * 0.5f * (1.0f + tanhf(sqrt_param * (x + mul_param * x * x * x)));
}

/*
Bias + GELU quantization kernel.
*/
template <int UNROLL>
__global__ void fused_bias_gelu_quantization(int8_t* __restrict__ output_data,
                                             float* __restrict__ scales,
                                             const __half* __restrict__ input_data,
                                             const __half* __restrict__ bias_data,
                                             int groups,
                                             int elems_per_group)
{
    // Conservative allocation, shared memory won't be an occupancy limiter though
    __shared__ float max_buffer[act_quant::num_warps];

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<act_quant::warp_size> warp =
        cg::tiled_partition<act_quant::warp_size>(tb);

    // Indexing offsets
    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * act_quant::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * act_quant::h_per_load;

    const __half* input_base = input_data + base_offset;
    const __half* bias_base = bias_data + elem_offset;

    __half2 local_buffer[UNROLL * act_quant::internal_unroll * act_quant::h2_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        __half2 bias_buffer[act_quant::internal_unroll * act_quant::h2_per_load];
        // Convenience helper, should resolve to register indices and not realize
        __half2* iteration_buffer =
            local_buffer + i * act_quant::internal_unroll * act_quant::h2_per_load;
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll; j++) {
            const int iteration = i * act_quant::internal_unroll + j;
            mem_access::load_global<act_quant::granularity>(
                iteration_buffer + j * act_quant::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
            mem_access::load_global<act_quant::granularity>(
                bias_buffer + j * act_quant::h2_per_load,
                bias_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }

        // TODO(cmikeh2): this might be faster with a tree reduce
        // but in general should mem bottlenecked so not a priority
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll * act_quant::h2_per_load; j++) {
            float2 val_f = __half22float2(iteration_buffer[j]);
            float2 bias_f = __half22float2(bias_buffer[j]);
            val_f.x = q_gelu(val_f.x + bias_f.x);
            val_f.y = q_gelu(val_f.y + bias_f.y);
            iteration_buffer[j] = __float22half2_rn(val_f);
        }
    }

    device_quantize<UNROLL>(
        local_buffer, scales, output_data, base_offset, elem_offset, stride, elems_per_group);
}

/*
LayerNorm quantization kernel.
Uses 2-pass layer normalization with __half precision underlying storage
(as opposed to leaving at float precision between when the layer norm completes
and the data is quantized)
*/
template <int UNROLL>
__global__ void fused_ln_quantization(int8_t* __restrict__ output_data,
                                      float* __restrict__ scales,
                                      const __half* __restrict__ input_data,
                                      const __half* __restrict__ gamma,
                                      const __half* __restrict__ beta,
                                      float epsilon,
                                      int groups,
                                      int elems_per_group)
{
    // Conservative allocation, shared memory won't be an occupancy limiter though
    // Use three buffers since its free and it eliminates some synchronizations
    __shared__ float mean_buffer[act_quant::num_warps];
    __shared__ float var_buffer[act_quant::num_warps];
    __shared__ float max_buffer[act_quant::num_warps];

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<act_quant::warp_size> warp =
        cg::tiled_partition<act_quant::warp_size>(tb);

    // Indexing offsets
    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * act_quant::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * act_quant::h_per_load;

    float2 thread_sum_mean_f2 = {0.f, 0.f};

    const __half* input_base = input_data + base_offset;

    constexpr int elems_per_thread = UNROLL * act_quant::internal_unroll * act_quant::h2_per_load;
    __half2 local_buffer[elems_per_thread];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize
        __half2* iteration_buffer =
            local_buffer + i * act_quant::internal_unroll * act_quant::h2_per_load;
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll; j++) {
            const int iteration = i * act_quant::internal_unroll + j;
            mem_access::load_global<act_quant::granularity>(
                iteration_buffer + j * act_quant::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }

        // TODO(cmikeh2): this might be faster with a tree reduce
        // but in general should mem bottlenecked so not a priority
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll * act_quant::h2_per_load; j++) {
            float2 val_f = __half22float2(iteration_buffer[j]);
            thread_sum_mean_f2.x += val_f.x;
            thread_sum_mean_f2.y += val_f.y;
        }
    }

    const float partial_mean = thread_sum_mean_f2.x + thread_sum_mean_f2.y;
    const float mean = reduce_sum(tb, warp, mean_buffer, partial_mean) / elems_per_group;

    float2 thread_sum_var_f2 = {0.f, 0.f};

#pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        float2 temp_up_cast = __half22float2(local_buffer[i]);
        thread_sum_var_f2.x += temp_up_cast.x;
        thread_sum_var_f2.y += temp_up_cast.y;
    }

    const float partial_var = thread_sum_var_f2.x + thread_sum_var_f2.y;
    const float variance = reduce_sum(tb, warp, var_buffer, partial_var) / elems_per_group;
    const float denom = __frsqrt_rn(variance + epsilon);

    float2 zeros = {0.f, 0.f};
    __half2 thread_max_h2 = __float22half2_rn(zeros);

    const __half* gamma_base = gamma + elem_offset;
    const __half* beta_base = beta + elem_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * act_quant::internal_unroll; i++) {
        __half2 gamma_local[act_quant::h2_per_load], beta_local[act_quant::h2_per_load];

        mem_access::load_global<act_quant::granularity>(
            gamma_local, gamma_base + i * stride, elem_offset + i * stride < elems_per_group);

        mem_access::load_global<act_quant::granularity>(
            beta_local, beta_base + i * stride, elem_offset + i * stride < elems_per_group);

        __half2* iteration_buffer = local_buffer + i * act_quant::h2_per_load;
        for (int j = 0; j < act_quant::h2_per_load; j++) {
            float2 up_cast = __half22float2(iteration_buffer[j]);
            up_cast.x = (up_cast.x - mean) * denom;
            up_cast.y = (up_cast.y - mean) * denom;
            up_cast.x = fmaf(up_cast.x, gamma_local[i].x, beta_local[i].x);
            up_cast.y = fmaf(up_cast.y, gamma_local[i].y, beta_local[i].y);
            iteration_buffer[j] = __float22half2_rn(up_cast);
        }
    }

    device_quantize<UNROLL>(
        local_buffer, scales, output_data, base_offset, elem_offset, stride, elems_per_group);
}
template <int UNROLL>
__global__ void fused_ln_quantization(int8_t* __restrict__ output_data,
                                      float* __restrict__ scales,
                                      __half* __restrict__ input_data,
                                      const __half* __restrict__ residual_data,
                                      const __half* __restrict__ bias_data,
                                      const __half* __restrict__ gamma,
                                      const __half* __restrict__ beta,
                                      float epsilon,
                                      int groups,
                                      int elems_per_group)
{
    // Conservative allocation, shared memory won't be an occupancy limiter though
    // Use three buffers since its free and it eliminates some synchronizations
    __shared__ float mean_buffer[act_quant::num_warps];
    __shared__ float var_buffer[act_quant::num_warps];
    __shared__ float max_buffer[act_quant::num_warps];

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<act_quant::warp_size> warp =
        cg::tiled_partition<act_quant::warp_size>(tb);

    // Indexing offsets
    const int block_offset = tb.group_index().x * elems_per_group;
    const int elem_offset = tb.thread_index().x * act_quant::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * act_quant::h_per_load;

    float2 thread_sum_mean_f2 = {0.f, 0.f};

    __half* input_base = input_data + base_offset;
    const __half* residual_base = residual_data + base_offset;
    const __half* bias_base = bias_data + elem_offset;

    constexpr int elems_per_thread = UNROLL * act_quant::internal_unroll * act_quant::h2_per_load;
    __half2 local_buffer[elems_per_thread];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        // Convenience helper, should resolve to register indices and not realize
        __half2* iteration_buffer =
            local_buffer + i * act_quant::internal_unroll * act_quant::h2_per_load;
        __half2 bias_buffer[act_quant::internal_unroll * act_quant::h2_per_load];
        // Convenience helper, should resolve to register indices and not realize
        __half2 residual_buffer[act_quant::internal_unroll * act_quant::h2_per_load];
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll; j++) {
            const int iteration = i * act_quant::internal_unroll + j;
            mem_access::load_global<act_quant::granularity>(
                iteration_buffer + j * act_quant::h2_per_load,
                input_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
            mem_access::load_global<act_quant::granularity>(
                residual_buffer + j * act_quant::h2_per_load,
                residual_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
            mem_access::load_global<act_quant::granularity>(
                bias_buffer + j * act_quant::h2_per_load,
                bias_base + iteration * stride,
                elem_offset + iteration * stride < elems_per_group);
        }

        // TODO(cmikeh2): this might be faster with a tree reduce
        // but in general should mem bottlenecked so not a priority
#pragma unroll
        for (int j = 0; j < act_quant::internal_unroll * act_quant::h2_per_load; j++) {
            float2 val_f = __half22float2(iteration_buffer[j]);
            float2 res_f = __half22float2(residual_buffer[j]);
            float2 bias_f = __half22float2(bias_buffer[j]);
            val_f.x = (val_f.x + res_f.x + bias_f.x);
            val_f.y = (val_f.y + res_f.y + bias_f.y);
            thread_sum_mean_f2.x += val_f.x;
            thread_sum_mean_f2.y += val_f.y;

            iteration_buffer[j] = __float22half2_rn(val_f);
        }
    }

    const float partial_mean = thread_sum_mean_f2.x + thread_sum_mean_f2.y;
    const float mean = reduce_sum(tb, warp, mean_buffer, partial_mean) / elems_per_group;

    float2 thread_sum_var_f2 = {0.f, 0.f};

#pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        float2 temp_up_cast = __half22float2(local_buffer[i]);
        thread_sum_var_f2.x += temp_up_cast.x;
        thread_sum_var_f2.y += temp_up_cast.y;
    }

    const float partial_var = thread_sum_var_f2.x + thread_sum_var_f2.y;
    const float variance = reduce_sum(tb, warp, var_buffer, partial_var) / elems_per_group;
    const float denom = __frsqrt_rn(variance + epsilon);

    float2 zeros = {0.f, 0.f};
    __half2 thread_max_h2 = __float22half2_rn(zeros);

    const __half* gamma_base = gamma + elem_offset;
    const __half* beta_base = beta + elem_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * act_quant::internal_unroll; i++) {
        __half2 gamma_local[act_quant::h2_per_load], beta_local[act_quant::h2_per_load];

        mem_access::load_global<act_quant::granularity>(
            gamma_local, gamma_base + i * stride, elem_offset + i * stride < elems_per_group);

        mem_access::load_global<act_quant::granularity>(
            beta_local, beta_base + i * stride, elem_offset + i * stride < elems_per_group);

        __half2* iteration_buffer = local_buffer + i * act_quant::h2_per_load;
        const int iteration = (i / UNROLL) * act_quant::internal_unroll + (i % UNROLL);
        for (int j = 0; j < act_quant::h2_per_load; j++) {
            float2 up_cast = __half22float2(iteration_buffer[j]);
            up_cast.x = (up_cast.x - mean) * denom;
            up_cast.y = (up_cast.y - mean) * denom;
            up_cast.x = fmaf(up_cast.x, gamma_local[i].x, beta_local[i].x);
            up_cast.y = fmaf(up_cast.y, gamma_local[i].y, beta_local[i].y);
            iteration_buffer[j] = __float22half2_rn(up_cast);
        }
        mem_access::store_global<act_quant::granularity>(input_base + iteration * stride,
                                                         iteration_buffer);
    }

    device_quantize<UNROLL>(
        local_buffer, scales, output_data, base_offset, elem_offset, stride, elems_per_group);
}

/********* Launcher methods ***********/

int32_t round_to_32(int32_t raw_value) { return (((raw_value - 1) >> 5) + 1) << 5; }

#define LAUNCH_ACTIVATION_QUANT(unroll_factor) \
    activation_quantization<unroll_factor>     \
        <<<grid, block, 0, stream>>>(output_data, scales, input_data, groups, elems_per_group);

void launch_act_quant(int8_t* output_data,
                      float* scales,
                      const __half* input_data,
                      int groups,
                      int elems_per_group,
                      cudaStream_t stream)
{
    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads =
        round_to_32((elems_per_group + act_quant::h_per_step - 1) / act_quant::h_per_step);
    const int threads = (one_step_threads < act_quant::threads) ? one_step_threads
                                                                : act_quant::threads;

    dim3 block(threads);
    dim3 grid(groups);

    const int elems_per_step = threads * act_quant::h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (external_unroll == 1) {
        // 0 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_ACTIVATION_QUANT(1);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_ACTIVATION_QUANT(2);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_ACTIVATION_QUANT(3);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_ACTIVATION_QUANT(4);
    }
}

#define LAUNCH_GELU_QUANT(unroll_factor)                                     \
    fused_bias_gelu_quantization<unroll_factor><<<grid, block, 0, stream>>>( \
        output_data, scales, input_data, bias_data, groups, elems_per_group);

void launch_gelu_quant(int8_t* output_data,
                       float* scales,
                       const __half* input_data,
                       const __half* bias_data,
                       int groups,
                       int elems_per_group,
                       cudaStream_t stream)
{
    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads =
        round_to_32((elems_per_group + act_quant::h_per_step - 1) / act_quant::h_per_step);
    const int threads = (one_step_threads < act_quant::threads) ? one_step_threads
                                                                : act_quant::threads;

    dim3 block(threads);
    dim3 grid(groups);

    const int elems_per_step = threads * act_quant::h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (external_unroll == 1) {
        // 0 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_GELU_QUANT(1);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_GELU_QUANT(2);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_GELU_QUANT(3);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_GELU_QUANT(4);
    }
}

#define LAUNCH_LN_QUANT(unroll_factor)                                \
    fused_ln_quantization<unroll_factor><<<grid, block, 0, stream>>>( \
        output_data, scales, input_data, gamma, beta, epsilon, groups, elems_per_group);

void launch_ln_quant(int8_t* output_data,
                     float* scales,
                     const __half* input_data,
                     const __half* gamma,
                     const __half* beta,
                     float epsilon,
                     int groups,
                     int elems_per_group,
                     cudaStream_t stream)
{
    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads =
        round_to_32((elems_per_group + act_quant::h_per_step - 1) / act_quant::h_per_step);
    const int threads = (one_step_threads < act_quant::threads) ? one_step_threads
                                                                : act_quant::threads;

    dim3 block(threads);
    dim3 grid(groups);

    const int elems_per_step = threads * act_quant::h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (external_unroll == 1) {
        // 0 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_LN_QUANT(1);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_LN_QUANT(2);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_LN_QUANT(3);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_LN_QUANT(4);
    }
}

#define LAUNCH_LN_RES_QUANT(unroll_factor)                                          \
    fused_ln_quantization<unroll_factor><<<grid, block, 0, stream>>>(output_data,   \
                                                                     scales,        \
                                                                     input_data,    \
                                                                     residual_data, \
                                                                     bias_data,     \
                                                                     gamma,         \
                                                                     beta,          \
                                                                     epsilon,       \
                                                                     groups,        \
                                                                     elems_per_group);

void launch_ln_quant(int8_t* output_data,
                     float* scales,
                     __half* input_data,
                     const __half* residual_data,
                     const __half* bias_data,
                     const __half* gamma,
                     const __half* beta,
                     float epsilon,
                     int groups,
                     int elems_per_group,
                     cudaStream_t stream)
{
    // Scheduling concern: may be slightly faster for some inputs to assign multiple stages of
    // warp-sized blocks rather than stepping up to 64/96 threads
    const int one_step_threads =
        round_to_32((elems_per_group + act_quant::h_per_step - 1) / act_quant::h_per_step);
    const int threads = (one_step_threads < act_quant::threads) ? one_step_threads
                                                                : act_quant::threads;

    dim3 block(threads);
    dim3 grid(groups);

    const int elems_per_step = threads * act_quant::h_per_step;
    const int external_unroll = (elems_per_group + elems_per_step - 1) / elems_per_step;

    if (external_unroll == 1) {
        // 0 - 4096 elems
        // (this can launch with 1-7 warps as well)
        LAUNCH_LN_RES_QUANT(1);
    } else if (external_unroll == 2) {
        // 4097 - 8192 elems
        LAUNCH_LN_RES_QUANT(2);
    } else if (external_unroll == 3) {
        // 8193 - 12288 elems
        LAUNCH_LN_RES_QUANT(3);
    } else if (external_unroll == 4) {
        // 12289 - 16384 elems
        LAUNCH_LN_RES_QUANT(4);
    }
}
