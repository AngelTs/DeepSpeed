#include <limits>
#include "custom_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define NORM_REG (MAX_REGISTERS)

namespace cg = cooperative_groups;

__global__ void fused_bias_residual_layer_norm(float* output,
                                               const float* vals,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               int row_stride)
{
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float inp_reg[NORM_REG];

    int k = 0;
    float sum = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k] = vals[input_id + row * row_stride];
        sum += inp_reg[k++];
        input_id += iteration_stride;
    }

    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);

    __shared__ float shr[MAX_WARP_NUM];

    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);

    float mean = sum / (row_stride);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f] -= mean;
        sum += inp_reg[f] * inp_reg[f];
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * sum;
        inp_reg[f] = inp_reg[f] * gamma[out_id] + beta[out_id];
        output[out_id + row * row_stride] = inp_reg[f];
    }
}

__global__ void fused_bias_residual_layer_norm(__half* output,
                                               const __half* vals,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    __half2 inp_reg[NORM_REG];

    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    __half2* out_cast = reinterpret_cast<__half2*>(output);

    int k = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k++] = vals_cast[input_id + row * row_stride];
        input_id += iteration_stride;
    }
    float sum = 0;
    for (int f = k - 1; f >= 0; f--) {
        float2 inp_f = __half22float2(inp_reg[f]);
        sum += inp_f.x + inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        float2 inp_f = __half22float2(inp_reg[f]);
        inp_f.x -= mean;
        inp_f.y -= mean;
        inp_reg[f] = __float22half2_rn(inp_f);
        sum += inp_f.x * inp_f.x;
        sum += inp_f.y * inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    __half2 variance_h = __float2half2_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * variance_h;
        inp_reg[f] = inp_reg[f] * gamma_cast[out_id] + beta_cast[out_id];
        out_cast[out_id + row * row_stride] = inp_reg[f];
    }
#endif
}

template <typename T>
void launch_layer_norm(T* out,
                       T* vals,
                       const T* gamma,
                       const T* beta,
                       float epsilon,
                       int batch_size,
                       int hidden_dim,
                       cudaStream_t stream);

template <>
void launch_layer_norm<float>(float* out,
                              float* vals,
                              const float* gamma,
                              const float* beta,
                              float epsilon,
                              int batch_size,
                              int hidden_dim,
                              cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        out, vals, gamma, beta, epsilon, hidden_dim);
}

template <>
void launch_layer_norm<__half>(__half* out,
                               __half* vals,
                               const __half* gamma,
                               const __half* beta,
                               float epsilon,
                               int batch_size,
                               int hidden_dim,
                               cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_bias_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(
        out, vals, gamma, beta, epsilon, hidden_dim / 2);
}

__global__ void fused_residual_layer_norm(float* norm,
                                          float* res_add,
                                          float* vals,
                                          float* residual,
                                          const float* bias,
                                          const float* gamma,
                                          const float* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN,
                                          bool mlp_after_attn)
{
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float inp_reg[NORM_REG];

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = vals[input_id + row * row_stride];
        float res_f = (residual[input_id + row * row_stride]);
        float bias_f = (bias[input_id]);
        if (mlp_after_attn) inp_reg[k] += res_f + bias_f;
        // if (preLN) res_add[input_id + row * row_stride] = inp_reg[k];
        sum += inp_reg[k++];
        input_id += iteration_stride;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);

    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f] -= mean;
        sum += inp_reg[f] * inp_reg[f];
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();

    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();

    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride);
    sum += epsilon;
    sum = __frsqrt_rn(sum);

    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * sum;
        inp_reg[f] = inp_reg[f] * gamma[out_id] + beta[out_id];
        norm[out_id + row * row_stride] = inp_reg[f];
    }
}

__global__ void fused_residual_layer_norm(__half* norm,
                                          __half* res_add,
                                          __half* vals,
                                          __half* residual,
                                          const __half* bias,
                                          const __half* gamma,
                                          const __half* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN,
                                          bool mlp_after_attn)
{
#ifdef HALF_PRECISION_AVAILABLE
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    __half2 inp_reg[NORM_REG];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    __half2* norm_cast = reinterpret_cast<__half2*>(norm);
    __half2* res_add_cast = reinterpret_cast<__half2*>(res_add);
    __half2* residual_cast = reinterpret_cast<__half2*>(residual);
    const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = vals_cast[input_id + row * row_stride];
        float2 inp_f = __half22float2(inp_reg[k]);
        float2 res_f = __half22float2(residual_cast[input_id + row * row_stride]);
        float2 bias_f = __half22float2(bias_cast[input_id]);
        if (mlp_after_attn) {
            inp_f.x += res_f.x + bias_f.x;
            inp_f.y += res_f.y + bias_f.y;
        }
        inp_reg[k] = __float22half2_rn(inp_f);
        // if (preLN) res_add_cast[input_id + row * row_stride] = __float22half2_rn(res_f);
        // //inp_reg[k];
        sum += inp_f.x + inp_f.y;
        input_id += iteration_stride;
        k++;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        float2 inp_f = __half22float2(inp_reg[f]);
        inp_f.x -= mean;
        inp_f.y -= mean;
        inp_reg[f] = __float22half2_rn(inp_f);
        sum += inp_f.x * inp_f.x;
        sum += inp_f.y * inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    __half2 variance_h = __float2half2_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f] = inp_reg[f] * variance_h;
        inp_reg[f] = inp_reg[f] * gamma_cast[out_id] + beta_cast[out_id];
        norm_cast[out_id + row * row_stride] = inp_reg[f];
    }
#endif
}

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

template <>
void launch_residual_layer_norm<float>(float* norm,
                                       float* res_add,
                                       float* vals,
                                       float* residual,
                                       const float* bias,
                                       const float* gamma,
                                       const float* beta,
                                       float epsilon,
                                       int batch_size,
                                       int hidden_dim,
                                       bool preLN,
                                       bool mlp_after_attn,
                                       cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);

    dim3 block_dim(threads);

    fused_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(norm,
                                                                  res_add,
                                                                  vals,
                                                                  residual,
                                                                  bias,
                                                                  gamma,
                                                                  beta,
                                                                  epsilon,
                                                                  hidden_dim,
                                                                  preLN,
                                                                  mlp_after_attn);
}

template <>
void launch_residual_layer_norm<__half>(__half* norm,
                                        __half* res_add,
                                        __half* vals,
                                        __half* residual,
                                        const __half* bias,
                                        const __half* gamma,
                                        const __half* beta,
                                        float epsilon,
                                        int batch_size,
                                        int hidden_dim,
                                        bool preLN,
                                        bool mlp_after_attn,
                                        cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_residual_layer_norm<<<grid_dim, block_dim, 0, stream>>>(norm,
                                                                  res_add,
                                                                  vals,
                                                                  residual,
                                                                  bias,
                                                                  gamma,
                                                                  beta,
                                                                  epsilon,
                                                                  hidden_dim / 2,
                                                                  preLN,
                                                                  mlp_after_attn);
}


__device__ void quantize_kernel_norm(float2* data,
                                unsigned cnt,
                                int8_t* vals_int,
                                float* q_scale_d,
                                int num_bits,
                                int group_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half* vals_int_cast = reinterpret_cast<__half*>(vals_int);

    float max = -10000.0;
    int bid = blockIdx.x;
    unsigned group_index;
    for(int i = 0;i < cnt;i++){
        if (fabsf(data[i].x) > max) max = fabsf(data[i].x);
        if (fabsf(data[i].y) > max) max = fabsf(data[i].y);
    }

    #pragma unroll
    for (int i = 1; i < WARP_SIZE; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (temp > max) max = temp;
    }
    __shared__ float partialMax[WARP_SIZE];

    if (lane == 0) partialMax[gid] = max;

    b.sync();

    max = partialMax[lane];

    b.sync();

    #pragma unroll
    for (int i = 1; i < warp_num; i <<= 1) {
        auto temp = g.shfl_xor(max, i);
        if (temp > max) max = temp;
    }
    max = g.shfl(max, 0);

    //float q_scale = (1 << num_bits) / (2 * max + 1e-5);
    float q_scale = (2 * max) / (1 << num_bits);

    group_index = threadIdx.x + bid * group_size;
    for(int i = 0;i < cnt;i++){
        __half q_data_int;
        int8_t* q_data_8 = reinterpret_cast<int8_t*>(&q_data_int);
        int32_t data_f[2];
        data_f[0] = round(data[i].x / q_scale);
        data_f[1] = round(data[i].y / q_scale);
        //q_data_8[0] = !(data_f[0] & 0x80000000) && (data_f[0] & 0x00000080) ? 127 : data_f[0];
        //q_data_8[1] = !(data_f[1] & 0x80000000) && (data_f[1] & 0x00000080) ? 127 : data_f[1];
        q_data_8[0] = data_f[0] > 127 ? 127 : (data_f[0] < -128 ? -128 : data_f[0]);
        q_data_8[1] = data_f[1] > 127 ? 127 : (data_f[1] < -128 ? -128 : data_f[1]);
        vals_int_cast[group_index] = q_data_int;
        group_index += (blockDim.x);
    }
    if (threadIdx.x == 0) q_scale_d[blockIdx.x] = q_scale;
}
__global__ void fused_bias_residual_layer_norm_int8(int8_t* output,
                                               float* scales,
                                               const float* vals,
                                               const float* gamma,
                                               const float* beta,
                                               float epsilon,
                                               int row_stride,
                                               int num_rows)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = WARP_SIZE;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    id = id & 0x1f;
    int warp_num = iteration_stride >> 5;
    int row = blockIdx.x * warp_num + gid;

    float2 inp_reg[NORM_REG];

    const float2* vals_cast = reinterpret_cast<const float2*>(vals);

    if(row < num_rows){
        int k = 0;
        int input_id = id;
        while (input_id < row_stride) {
            inp_reg[k++] = vals_cast[input_id + row * row_stride];
            input_id += iteration_stride;
        }
        float sum = 0;
        for (int f = k - 1; f >= 0; f--) {
            float2 inp_f = inp_reg[f];
            sum += inp_f.x + inp_f.y;
        }
        for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
        __shared__ float shr[MAX_WARP_NUM];
        if (g.thread_rank() == 0) shr[gid] = sum;
        b.sync();
        if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        b.sync();
        for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
        sum = g.shfl(sum, 0);
        float mean = sum / (row_stride << 1);
        sum = 0.f;
        for (int f = 0; f < k; f++) {
            float2 inp_f = inp_reg[f];
            inp_f.x -= mean;
            inp_f.y -= mean;
            inp_reg[f] = inp_f;
            sum += inp_f.x * inp_f.x;
            sum += inp_f.y * inp_f.y;
        }
        for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
        if (g.thread_rank() == 0) shr[gid] = sum;
        b.sync();
        if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        b.sync();
        for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
        sum = g.shfl(sum, 0);
        sum /= (row_stride << 1);
        sum += epsilon;
        sum = __frsqrt_rn(sum);
        const float2* gamma_cast = reinterpret_cast<const float2*>(gamma);
        const float2* beta_cast = reinterpret_cast<const float2*>(beta);
        for (int f = 0; f < k; f++) {
            int out_id = f * iteration_stride + id;
            inp_reg[f].x = inp_reg[f].x * sum;
            inp_reg[f].y = inp_reg[f].y * sum;
            float2 gamma_reg = gamma_cast[out_id];
            float2 beta_reg = beta_cast[out_id];
            inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
            inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
            //out_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
        }
        quantize_kernel_norm(inp_reg,
            k,
            output,
            scales,
            8,
            row_stride);
    }
#endif
}
__global__ void fused_bias_residual_layer_norm_int8(int8_t* output,
                                               float* scales,
                                               const __half* vals,
                                               const __half* gamma,
                                               const __half* beta,
                                               float epsilon,
                                               int row_stride,
                                               int num_rows)
{
#if __CUDA_ARCH__ >= 700
/*    int iteration_stride = WARP_SIZE;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    id = id & 0x1f;
    int warp_num = iteration_stride >> 5;
    int row = blockIdx.x * warp_num + gid;

    float2 inp_reg[NORM_REG];

    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    if (row < num_rows){
        int k = 0;
        int input_id = id;
        while (input_id < row_stride) {
            inp_reg[k++] = vals_cast[input_id + row * row_stride];
            input_id += iteration_stride;
        }
        float sum = 0;
        for (int f = k - 1; f >= 0; f--) {
            __half2 *inp_f = (__half2*)&inp_reg[f];
            float2 sum1 = __half22float2(inp_f[0] + inp_f[1]);
            sum += sum1.x + sum1.y;
        }
        for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
        //__shared__ float shr[MAX_WARP_NUM];
        //if (g.thread_rank() == 0) shr[gid] = sum;
        //b.sync();
        //if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        //b.sync();
        //for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
        //sum = g.shfl(sum, 0);
        __half2 mean = __float2half2_rn(sum / (row_stride << 2));
        sum = 0.f;
        for (int f = 0; f < k; f++) {
            __half2 *inp_f = (__half2*)&inp_reg[f];
            //float2 inp_f = inp_reg[f];
            //inp_f.x -= mean;
            //inp_f.y -= mean;
            //inp_reg[f] = inp_f;
            inp_f[0] -= mean;
            inp_f[1] -= mean;
            float2 sum1 = __half22float2(inp_f[0] * inp_f[0]);
            float2 sum2 = __half22float2(inp_f[1] * inp_f[1]);
            sum1.x += sum1.y;
            sum2.x += sum2.y;
            sum += sum1.x + sum2.x;
            //sum += inp_f.x * inp_f.x;
            //sum += inp_f.y * inp_f.y;
        }
        for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
        //if (g.thread_rank() == 0) shr[gid] = sum;
        //b.sync();
        //if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        //b.sync();
        //for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
        //sum = g.shfl(sum, 0);
        sum /= (row_stride << 2);
        sum += epsilon;
        sum = __frsqrt_rn(sum);
        __half2 var_reg = __float2half2_rn(sum);
        const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
        const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
        for (int f = 0; f < k; f++) {
            int out_id = f * iteration_stride + id;
            __half2 gamma_reg = gamma_cast[out_id];
            __half2 beta_reg = beta_cast[out_id];
            __half2 *inp_f = (__half2*)&inp_reg[f];
            inp_f[0] = inp_f[0] * var_reg * gamma_reg + beta_reg;
            inp_f[1] = inp_f[1] * var_reg * gamma_reg + beta_reg;
            //inp_reg[f].x = inp_reg[f].x * sum;
            //inp_reg[f].y = inp_reg[f].y * sum;
            //float2 gamma_reg = gamma_cast[out_id];
            //float2 beta_reg = beta_cast[out_id];
            //inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
            //inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
            //out_cast[out_id + row * row_stride] = (inp_reg[f]);
        }
        quantize_kernel_norm_half(inp_reg,
            k,
            output,
            scales,
            8,
            row_stride,
            row);
    }*/
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float2 inp_reg[NORM_REG];

    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);

    int k = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k++] = __half22float2(vals_cast[input_id + row * row_stride]);
        input_id += iteration_stride;
    }
    float sum = 0;
    for (int f = k - 1; f >= 0; f--) {
        float2 inp_f = inp_reg[f];
        sum += inp_f.x + inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        float2 inp_f = inp_reg[f];
        inp_f.x -= mean;
        inp_f.y -= mean;
        inp_reg[f] = inp_f;
        sum += inp_f.x * inp_f.x;
        sum += inp_f.y * inp_f.y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f].x = inp_reg[f].x * sum;
        inp_reg[f].y = inp_reg[f].y * sum;
        float2 gamma_reg = __half22float2(gamma_cast[out_id]);
        float2 beta_reg = __half22float2(beta_cast[out_id]);
        inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
        inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
        //out_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    quantize_kernel_norm(inp_reg,
        k,
        output,
        scales,
        8,
        row_stride);
#endif
}

template<typename T>
void launch_layer_norm_int8(int8_t* out,
                            float* scales,
                            T* vals,
                            const T* gamma,
                            const T* beta,
                            float epsilon,
                            int batch_size,
                            int hidden_dim,
                            cudaStream_t stream);
template <>
void launch_layer_norm_int8<float>(int8_t* out,
                            float* scales,
                            float* vals,
                            const float* gamma,
                            const float* beta,
                            float epsilon,
                            int batch_size,
                            int hidden_dim,
                            cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);
    //dim3 grid_dim((batch_size-1) / 32 + 1);
    //dim3 block_dim(threads);

    fused_bias_residual_layer_norm_int8<<<grid_dim, block_dim, 0, stream>>>(
        out, scales, vals, gamma, beta, epsilon, hidden_dim / 2, batch_size);
}
template <>
void launch_layer_norm_int8<__half>(int8_t* out,
                            float* scales,
                            __half* vals,
                            const __half* gamma,
                            const __half* beta,
                            float epsilon,
                            int batch_size,
                            int hidden_dim,
                            cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);
    //dim3 grid_dim((batch_size-1) / 32 + 1);
    //dim3 block_dim(threads);

    fused_bias_residual_layer_norm_int8<<<grid_dim, block_dim, 0, stream>>>(
        out, scales, vals, gamma, beta, epsilon, hidden_dim / 2, batch_size);
}


__global__ void fused_residual_layer_norm_int8(int8_t* res_add,
                                          float* scales,
                                          __half* vals,
                                          __half* residual,
                                          const __half* bias,
                                          const __half* gamma,
                                          const __half* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN,
                                          int mp_size)
{
#if __CUDA_ARCH__ >= 700
/*    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float2 inp_reg[NORM_REG];

    float2* vals_cast = reinterpret_cast<float2*>(vals);
    //__half2* res_add_cast = reinterpret_cast<__half2*>(res_add);
    float2* residual_cast = reinterpret_cast<float2*>(residual);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = (vals_cast[input_id + row * row_stride]);
        
        __half2 *inp_f = (__half2*)&inp_reg[k];
        float2 res_f = residual_cast[input_id + row * row_stride];
        float2 bias_f = bias_cast[input_id];
        __half2 *res_h = (__half2*)&res_f;
        __half2 *bias_h = (__half2*)&bias_f;
        
        inp_f[0] += res_h[0] + bias_h[0];
        inp_f[1] += res_h[1] + bias_h[1];

        float2 sum1 = __half22float2(inp_f[0] + inp_f[1]);
        // if (preLN) res_add_cast[input_id + row * row_stride] = inp_reg[k];
        sum += sum1.x + sum1.y;
        input_id += iteration_stride;
        k++;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    __half2 mean = __float2half2_rn(sum / (row_stride << 1));
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        __half2 *inp_f = (__half2*)&inp_reg[f];
        //float2 inp_f = inp_reg[f];
        //inp_f.x -= mean;
        //inp_f.y -= mean;
        //inp_reg[f] = inp_f;
        inp_f[0] -= mean;
        inp_f[1] -= mean;
        float2 sum1 = __half22float2(inp_f[0] * inp_f[0]);
        float2 sum2 = __half22float2(inp_f[1] * inp_f[1]);
        sum1.x += sum1.y;
        sum2.x += sum2.y;
        sum += sum1.x + sum2.x;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    __half2 var_reg = __float2half2_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        __half2 gamma_reg = gamma_cast[out_id];
        __half2 beta_reg = beta_cast[out_id];
        __half2 *inp_f = (__half2*)&inp_reg[f];
        inp_f[0] = inp_f[0] * var_reg * gamma_reg + beta_reg;
        inp_f[1] = inp_f[1] * var_reg * gamma_reg + beta_reg;
    }
    quantize_kernel111(inp_reg,
        k,
        res_add,
        scales,
        8,
        row_stride);*/
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float2 inp_reg[NORM_REG];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    //__half2* res_add_cast = reinterpret_cast<__half2*>(res_add);
    __half2* residual_cast = reinterpret_cast<__half2*>(residual);
    const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = __half22float2(vals_cast[input_id + row * row_stride]);
        float2 res_f = __half22float2(residual_cast[input_id + row * row_stride]);
        float2 bias_f = __half22float2(bias_cast[input_id]);
        inp_reg[k].x += res_f.x + bias_f.x;
        inp_reg[k].y += res_f.y + bias_f.y;

        // if (preLN) res_add_cast[input_id + row * row_stride] = inp_reg[k];
        sum += inp_reg[k].x + inp_reg[k].y;
        input_id += iteration_stride;
        k++;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f].x -= mean;
        inp_reg[f].y -= mean;
        sum += inp_reg[f].x * inp_reg[f].x;
        sum += inp_reg[f].y * inp_reg[f].y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f].x = inp_reg[f].x * sum;
        inp_reg[f].y = inp_reg[f].y * sum;
        float2 gamma_reg = __half22float2(gamma_cast[out_id]);
        float2 beta_reg  = __half22float2(beta_cast[out_id]);
        inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
        inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
        //res_add_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    quantize_kernel_norm(inp_reg,
        k,
        res_add,
        scales,
        8,
        row_stride);
#endif
}

__global__ void fused_residual_layer_norm_int8(int8_t* res_add,
                                          float* scales,
                                          float* vals,
                                          float* residual,
                                          const float* bias,
                                          const float* gamma,
                                          const float* beta,
                                          float epsilon,
                                          int row_stride,
                                          bool preLN,
                                          int mp_size)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = blockDim.x;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int row = blockIdx.x;
    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    float2 inp_reg[NORM_REG];

    __half2* vals_cast = reinterpret_cast<__half2*>(vals);
    //__half2* res_add_cast = reinterpret_cast<__half2*>(res_add);
    float2* residual_cast = reinterpret_cast<float2*>(residual);
    const float2* bias_cast = reinterpret_cast<const float2*>(bias);

    int k = 0;
    int input_id = id;

    float sum = 0;
    while (input_id < row_stride) {
        inp_reg[k] = __half22float2(vals_cast[input_id + row * row_stride]);
        float2 res_f = residual_cast[input_id + row * row_stride];
        float2 bias_f = bias_cast[input_id];
        inp_reg[k].x += res_f.x + (bias_f.x * mp_size);
        inp_reg[k].y += res_f.y + (bias_f.y * mp_size);

        // if (preLN) res_add_cast[input_id + row * row_stride] = inp_reg[k];
        sum += inp_reg[k].x + inp_reg[k].y;
        input_id += iteration_stride;
        k++;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    __shared__ float shr[MAX_WARP_NUM];
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    float mean = sum / (row_stride << 1);
    sum = 0.f;
    for (int f = 0; f < k; f++) {
        inp_reg[f].x -= mean;
        inp_reg[f].y -= mean;
        sum += inp_reg[f].x * inp_reg[f].x;
        sum += inp_reg[f].y * inp_reg[f].y;
    }
    for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
    if (g.thread_rank() == 0) shr[gid] = sum;
    b.sync();
    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
    b.sync();
    for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
    sum = g.shfl(sum, 0);
    sum /= (row_stride << 1);
    sum += epsilon;
    sum = __frsqrt_rn(sum);
    const float2* gamma_cast = reinterpret_cast<const float2*>(gamma);
    const float2* beta_cast = reinterpret_cast<const float2*>(beta);
    for (int f = 0; f < k; f++) {
        int out_id = f * iteration_stride + id;
        inp_reg[f].x = inp_reg[f].x * sum;
        inp_reg[f].y = inp_reg[f].y * sum;
        float2 gamma_reg = gamma_cast[out_id];
        float2 beta_reg = beta_cast[out_id];
        inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
        inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
        //res_add_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    //quantize_kernel_norm(inp_reg,
    //    k,
    //    res_add,
    //    scales,
    //    8,
    //    row_stride);
#endif
}

template<typename T>
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
                                     cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_residual_layer_norm_int8<<<grid_dim, block_dim, 0, stream>>>(
        res_add, scales, vals, residual, bias, gamma, beta, epsilon, hidden_dim / 2, preLN, mp_size);
}


template void launch_residual_layer_norm_int8<__half>(int8_t* res_add,
    float* scales,
    __half* vals,
    __half* residual,
    const __half* bias,
    const __half* gamma,
    const __half* beta,
    float epsilon,
    int batch_size,
    int hidden_dim,
    bool preLN,
    int mp_size,
    cudaStream_t stream);
template void launch_residual_layer_norm_int8<float>(int8_t* res_add,
                                     float* scales,
                                     float* vals,
                                     float* residual,
                                     const float* bias,
                                     const float* gamma,
                                     const float* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     bool preLN,
                                     int mp_size,
                                     cudaStream_t stream);


                                     __global__ void fused_residual_layer_norm1(float* norm,
                                        float* vals,
                                        float* residual,
                                        const float* bias,
                                        const float* gamma,
                                        const float* beta,
                                        float epsilon,
                                        int row_stride)
{
  int iteration_stride = blockDim.x;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  int row = blockIdx.x;
  int id = threadIdx.x;
  int gid = id >> 5;
  int warp_num = iteration_stride >> 5;

  float inp_reg[NORM_REG];

  int k = 0;
  int input_id = id;

  float sum = 0;
  while (input_id < row_stride) {
      inp_reg[k] = vals[input_id + row * row_stride];
      float res_f = (residual[input_id + row * row_stride]);
      float bias_f = (bias[input_id]);
      inp_reg[k] += res_f + bias_f;
      sum += inp_reg[k++];
      input_id += iteration_stride;
  }
  for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);

  __shared__ float shr[MAX_WARP_NUM];
  if (g.thread_rank() == 0) shr[gid] = sum;
  b.sync();

  if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
  b.sync();

  for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
  sum = g.shfl(sum, 0);
  float mean = sum / (row_stride);
  sum = 0.f;
  for (int f = 0; f < k; f++) {
      inp_reg[f] -= mean;
      sum += inp_reg[f] * inp_reg[f];
  }
  for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
  if (g.thread_rank() == 0) shr[gid] = sum;
  b.sync();

  if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
  b.sync();

  for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
  sum = g.shfl(sum, 0);
  sum /= (row_stride);
  sum += epsilon;
  sum = __frsqrt_rn(sum);

  for (int f = 0; f < k; f++) {
      int out_id = f * iteration_stride + id;
      inp_reg[f] = inp_reg[f] * sum;
      inp_reg[f] = inp_reg[f] * gamma[out_id] + beta[out_id];
      norm[out_id + row * row_stride] = inp_reg[f];
  }
}

__global__ void fused_residual_layer_norm1(__half* norm,
                                        __half* vals,
                                        __half* residual,
                                        const __half* bias,
                                        const __half* gamma,
                                        const __half* beta,
                                        float epsilon,
                                        int row_stride)
{
#ifdef HALF_PRECISION_AVAILABLE
  int iteration_stride = blockDim.x;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

  int row = blockIdx.x;
  int id = threadIdx.x;
  int gid = id >> 5;
  int warp_num = iteration_stride >> 5;

  float2 inp_reg[16];

  __half2* vals_cast = reinterpret_cast<__half2*>(vals);
  __half2* norm_cast = reinterpret_cast<__half2*>(norm);
  __half2* residual_cast = reinterpret_cast<__half2*>(residual);
  const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);

  int k = 0;
  int input_id = id;

  float sum = 0;
  while (input_id < row_stride) {
      auto tmp = vals_cast[input_id + row * row_stride];
      inp_reg[k] = __half22float2(tmp);
      float2 res_f = __half22float2(residual_cast[input_id + row * row_stride]);
      float2 bias_f = __half22float2(bias_cast[input_id]);
      inp_reg[k].x += res_f.x + bias_f.x;
      inp_reg[k].y += res_f.y + bias_f.y;

      sum += inp_reg[k].x + inp_reg[k].y;
      input_id += iteration_stride;
      k++;
  }
  for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
  __shared__ float shr[MAX_WARP_NUM];
  if (g.thread_rank() == 0) shr[gid] = sum;
  b.sync();
  if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
  b.sync();
  for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
  sum = g.shfl(sum, 0);
  float mean = sum / (row_stride << 1);
  sum = 0.f;
  for (int f = 0; f < k; f++) {
      inp_reg[f].x -= mean;
      inp_reg[f].y -= mean;
      sum += inp_reg[f].x * inp_reg[f].x;
      sum += inp_reg[f].y * inp_reg[f].y;
  }
  for (int i = 1; i < 32; i *= 2) sum += g.shfl_down(sum, i);
  if (g.thread_rank() == 0) shr[gid] = sum;
  b.sync();
  if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
  b.sync();
  for (int i = 1; i < (warp_num); i *= 2) sum += g.shfl_down(sum, i);
  sum = g.shfl(sum, 0);
  sum /= (row_stride << 1);
  sum += epsilon;
  sum = __frsqrt_rn(sum);

  const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
  const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
  for (int f = 0; f < k; f++) {
      int out_id = f * iteration_stride + id;
      inp_reg[f].x *= sum;
      inp_reg[f].y *= sum;
      auto tmp = __float22half2_rn(inp_reg[f]);
      tmp = tmp * gamma_cast[out_id] + beta_cast[out_id];
      norm_cast[out_id + row * row_stride] = tmp;
  }
#endif
}

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


template <>
void launch_residual_layer_norm1<float>(float* norm,
                                     float* vals,
                                     float* residual,
                                     const float* bias,
                                     const float* gamma,
                                     const float* beta,
                                     float epsilon,
                                     int batch_size,
                                     int hidden_dim,
                                     cudaStream_t stream)
{
  constexpr int threads = 1024;

  dim3 grid_dim(batch_size);

  dim3 block_dim(threads);

  fused_residual_layer_norm1<<<grid_dim, block_dim, 0, stream>>>(
      norm, vals, residual, bias, gamma, beta, epsilon, hidden_dim);
}

template <>
void launch_residual_layer_norm1<__half>(__half* norm,
                                      __half* vals,
                                      __half* residual,
                                      const __half* bias,
                                      const __half* gamma,
                                      const __half* beta,
                                      float epsilon,
                                      int batch_size,
                                      int hidden_dim,
                                      cudaStream_t stream)
{
  constexpr int threads = 1024;

  dim3 grid_dim(batch_size);
  dim3 block_dim(threads);

  fused_residual_layer_norm1<<<grid_dim, block_dim, 0, stream>>>(
      norm, vals, residual, bias, gamma, beta, epsilon, hidden_dim / 2);
}