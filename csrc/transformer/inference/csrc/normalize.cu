/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include <limits>
#include "inference_cuda_layers.h"

#ifndef __HIP_PLATFORM_HCC__
#include <cuda_profiler_api.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "layer_norm_utils.h"
#include "memory_access_utils.h"

#include "conversion_utils.h"
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
    for (int i = 0; i < cnt; i++) {
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

    // float q_scale = (1 << num_bits) / (2 * max + 1e-5);
    float q_scale = (2 * max) / (1 << num_bits);

    group_index = threadIdx.x + bid * group_size;
    for (int i = 0; i < cnt; i++) {
        __half q_data_int;
        int8_t* q_data_8 = reinterpret_cast<int8_t*>(&q_data_int);
        int32_t data_f[2];
        data_f[0] = round(data[i].x / q_scale);
        data_f[1] = round(data[i].y / q_scale);
        // q_data_8[0] = !(data_f[0] & 0x80000000) && (data_f[0] & 0x00000080) ? 127 : data_f[0];
        // q_data_8[1] = !(data_f[1] & 0x80000000) && (data_f[1] & 0x00000080) ? 127 : data_f[1];
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

    if (row < num_rows) {
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
            // out_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
        }
        quantize_kernel_norm(inp_reg, k, output, scales, 8, row_stride);
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
        // out_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    quantize_kernel_norm(inp_reg, k, output, scales, 8, row_stride);
#endif
}

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
    // dim3 grid_dim((batch_size-1) / 32 + 1);
    // dim3 block_dim(threads);

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
    // dim3 grid_dim((batch_size-1) / 32 + 1);
    // dim3 block_dim(threads);

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
        float2 beta_reg = __half22float2(beta_cast[out_id]);
        inp_reg[f].x = inp_reg[f].x * gamma_reg.x + beta_reg.x;
        inp_reg[f].y = inp_reg[f].y * gamma_reg.y + beta_reg.y;
        // res_add_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    quantize_kernel_norm(inp_reg, k, res_add, scales, 8, row_stride);
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
        // res_add_cast[out_id + row * row_stride] = __float22half2_rn(inp_reg[f]);
    }
    // quantize_kernel_norm(inp_reg,
    //    k,
    //    res_add,
    //    scales,
    //    8,
    //    row_stride);
#endif
}

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
                                     cudaStream_t stream)
{
    constexpr int threads = 1024;

    dim3 grid_dim(batch_size);
    dim3 block_dim(threads);

    fused_residual_layer_norm_int8<<<grid_dim, block_dim, 0, stream>>>(res_add,
                                                                       scales,
                                                                       vals,
                                                                       residual,
                                                                       bias,
                                                                       gamma,
                                                                       beta,
                                                                       epsilon,
                                                                       hidden_dim / 2,
                                                                       preLN,
                                                                       mp_size);
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

template <typename T, int UNROLL>
__global__ void fused_ln(T* output,
                         const T* vals,
                         const T* gamma,
                         const T* beta,
                         float epsilon,
                         int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    __shared__ float sum_buffer[ln::max_warps];
    __shared__ float var_buffer[ln::max_warps];

    // X-dimension of the block
    const int block_offset = tb.group_index().x * elems_per_row;
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float partial_sum = 0.f;

    const T* input_base = vals + base_offset;
    T local_buffer[UNROLL * ln::internal_unroll * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
        T* iteration_buffer = local_buffer + i * ln::internal_unroll * T_per_load;

#pragma unroll
        for (int j = 0; j < ln::internal_unroll; j++) {
            const int iteration = i * ln::internal_unroll + j;
            mem_access::load_global<ln::granularity>(
                iteration_buffer + j * T_per_load,
                input_base + iteration * stride,
                thread_offset + iteration * stride < elems_per_row);
        }

#pragma unroll
        for (int j = 0; j < ln::internal_unroll * T_per_load; j++) {
            partial_sum += conversion::to<float>(iteration_buffer[j]);
        }
    }

    const float sum = ln_sum_reduce<ln::max_warps>(tb, warp, sum_buffer, partial_sum);
    const float mean = sum / elems_per_row;

    float partial_mean_diff = 0.f;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                partial_mean_diff += diff * diff;
            }
        }
    }

    const float mean_diff = ln_sum_reduce<ln::max_warps>(tb, warp, var_buffer, partial_mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_FUSED_LN(unroll_factor) \
    fused_ln<T, unroll_factor>         \
        <<<grid, block, 0, stream>>>(output, vals, gamma, beta, epsilon, elems_per_row);

template <typename T>
void launch_fused_ln(T* output,
                     const T* vals,
                     const T* gamma,
                     const T* beta,
                     float epsilon,
                     int rows,
                     int elems_per_row,
                     cudaStream_t stream)
{
    constexpr int max_unroll = 4;

    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_LN(1);
    } else if (unroll == 2) {
        LAUNCH_FUSED_LN(2);
    } else if (unroll == 3) {
        LAUNCH_FUSED_LN(3);
    } else if (unroll == 4) {
        LAUNCH_FUSED_LN(4);
    }
}

template void launch_fused_ln(__half*,
                              const __half*,
                              const __half*,
                              const __half*,
                              float,
                              int,
                              int,
                              cudaStream_t);
template void
launch_fused_ln(float*, const float*, const float*, const float*, float, int, int, cudaStream_t);

template <typename T, int UNROLL>
__global__ void fused_residual_ln(T* output,
                                  const T* vals,
                                  const T* residual,
                                  const T* bias,
                                  const T* gamma,
                                  const T* beta,
                                  float epsilon,
                                  int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    __shared__ float sum_buffer[ln::max_warps];
    __shared__ float var_buffer[ln::max_warps];

    // X-dimension of the block
    const int block_offset = tb.group_index().x * elems_per_row;
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float partial_sum = 0.f;

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    const T* bias_base = bias + thread_offset;

    T local_buffer[UNROLL * ln::internal_unroll * T_per_load];

    // Unlike a vanilla layernorm, since we're fusing the two adds as well
    // an inner unroll seems to be less valuable. If anything, a double unroll
    // makes the most sense if we find we are having performance issues.
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        T residual_buffer[T_per_load];
        T bias_buffer[T_per_load];

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(residual_buffer,
                                                 residual_base + i * stride,
                                                 thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(
            bias_buffer, bias_base + i * stride, thread_offset + i * stride < elems_per_row);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            float res_up_cast = conversion::to<float>(residual_buffer[j]);
            float bias_up_cast = conversion::to<float>(bias_buffer[j]);
            vals_up_cast += res_up_cast + bias_up_cast;
            partial_sum += vals_up_cast;
            iteration_buffer[j] = conversion::to<T>(vals_up_cast);
        }
    }

    const float sum = ln_sum_reduce<ln::max_warps>(tb, warp, sum_buffer, partial_sum);
    const float mean = sum / elems_per_row;

    float partial_mean_diff = 0.f;
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                partial_mean_diff += diff * diff;
            }
        }
    }

    const float mean_diff = ln_sum_reduce<ln::max_warps>(tb, warp, var_buffer, partial_mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
}

#define LAUNCH_FUSED_RES_LN(unroll_factor)                           \
    fused_residual_ln<T, unroll_factor><<<grid, block, 0, stream>>>( \
        output, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

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
                              cudaStream_t stream)
{
    constexpr int max_unroll = 4;

    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_RES_LN(1);
    } else if (unroll == 2) {
        LAUNCH_FUSED_RES_LN(2);
    } else if (unroll == 3) {
        LAUNCH_FUSED_RES_LN(3);
    } else if (unroll == 4) {
        LAUNCH_FUSED_RES_LN(4);
    }
}

template void launch_fused_residual_ln(__half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       const __half*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);
template void launch_fused_residual_ln(float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       const float*,
                                       float,
                                       int,
                                       int,
                                       cudaStream_t);

namespace cg = cooperative_groups;

namespace act_quant {
constexpr int granularity = 16;
constexpr int h_per_load = granularity / sizeof(__half);
constexpr int h2_per_load = granularity / sizeof(__half2);

constexpr int threads = 256;
// BRITTLE
constexpr int warp_size = 32;
constexpr int num_warps = threads / warp_size;

constexpr int internal_unroll = ln::internal_unroll;
constexpr int h_per_step = h_per_load * internal_unroll;

// Currently hardcoded, can re-evaluate in the future
constexpr int q_bits = 8;
constexpr int q_range = 1 << q_bits;
}  // namespace act_quant

/*
Quantization reduction helper. Input is the max value seen by each thread,
returns the quantization scale. Inverse still needs to be stored to global
memory by the caller.
*/
template <int q_bits>
__device__ __forceinline__ float get_scale(cg::thread_block& tb,
                                           cg::thread_block_tile<act_quant::warp_size>& warp,
                                           float* max_buffer,
                                           float thread_max_arg)
{
    float thread_max_f = thread_max_arg;
    auto q_range = 1 << q_bits;
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

            const float quantization_scale = q_range / (2 * r_max);

            if (warp.thread_rank() == 0) { max_buffer[0] = quantization_scale; }
        }

        // Safe in the conditional since all threads will evaluate the if-statement identically
        tb.sync();

        return max_buffer[0];
    } else {
        // Otherwise broadcast from thread 0 and continue
        const float quantization_scale = q_range / (2 * thread_max_f);

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

    if constexpr (q_bits == 8) {
#pragma unroll
        for (int i = 0; i < elems; i++) {
            // TODO(cmikeh2): refactor to use conversion utils
            float data_f = __half2float(data[i]) * scale;
            int32_t data_i32 = __float2int_rn(data_f);
            data_i32 = min(max(data_i32, q_min), q_max);
            local_output[i] = (int8_t)data_i32;
        }

    } else if constexpr (q_bits == 4) {
#pragma unroll
        for (int i = 0; i < elems / 2; i++) {
            float data_f_1 = __half2float(data[2 * i]) * scale;
            float data_f_2 = __half2float(data[2 * i + 1]) * scale;
            int32_t data_i32_1 = __float2int_rn(data_f_1);
            int32_t data_i32_2 = __float2int_rn(data_f_2);
            int8_t data_i8_1 = (int8_t)min(max(data_i32_1, q_min), q_max);
            int8_t data_i8_2 = (int8_t)min(max(data_i32_2, q_min), q_max);
            auto data_i8 = int4x2_t{data_i8_2, data_i8_1};
            local_output[i] = *((int8_t*)(&data_i8));
        }
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

template <int UNROLL, int q_bits = 8>
__device__ void device_quantize(__half* local_buffer_h,
                                float* __restrict__ scales,
                                int8_t* __restrict__ output_data,
                                const int& base_offset,
                                const int& elem_offset,
                                const int& stride,
                                const int& elems_per_group)
{
    // Conservative allocation, shared memory won't be an occupancy limiter though
    __half2* local_buffer = reinterpret_cast<__half2*>(local_buffer_h);
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

    float q_scale = get_scale<q_bits>(tb, warp, max_buffer, thread_max_f);
    if (tb.thread_index().x == 0) scales[tb.group_index().x] = 1 / q_scale;

#pragma unroll
    for (int i = 0; i < UNROLL * act_quant::internal_unroll; i++) {
        if constexpr (q_bits == 8) {
            int8_t local_output[act_quant::h_per_load];
            quant_16_bytes<q_bits>(
                local_output, local_buffer + i * act_quant::h2_per_load, q_scale);
            if (elem_offset + i * stride < elems_per_group) {
                mem_access::store_global<act_quant::granularity / 2>(
                    output_data + base_offset + i * stride, local_output);
            }
        } else if constexpr (q_bits == 4) {
            int8_t local_output[act_quant::h_per_load/2];
            quant_16_bytes<q_bits>(
                local_output, local_buffer + i * act_quant::h2_per_load, q_scale);
            if (elem_offset + i * stride < elems_per_group) {
                mem_access::store_global<act_quant::granularity / 4>(
                    output_data + (base_offset + i * stride)/2, local_output);
            }
        }
    }
}

template <typename T, int UNROLL, int q_bits>
__global__ void fused_residual_ln_quant(T* output,
                                  int8_t* out_int8,
                                  float* scales,
                                  const T* vals,
                                  const T* residual,
                                  const T* bias,
                                  const T* gamma,
                                  const T* beta,
                                  float epsilon,
                                  int elems_per_row)
{
    constexpr int T_per_load = ln::granularity / sizeof(T);

    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    __shared__ float sum_buffer[ln::max_warps];
    __shared__ float var_buffer[ln::max_warps];

    // X-dimension of the block
    const int block_offset = tb.group_index().x * elems_per_row;
    const int thread_offset = tb.thread_index().x * T_per_load;
    const int base_offset = block_offset + thread_offset;
    const int stride = tb.size() * T_per_load;

    float partial_sum = 0.f;

    const T* input_base = vals + base_offset;
    const T* residual_base = residual + base_offset;
    const T* bias_base = bias + thread_offset;

    T local_buffer[UNROLL * ln::internal_unroll * T_per_load];

    // Unlike a vanilla layernorm, since we're fusing the two adds as well
    // an inner unroll seems to be less valuable. If anything, a double unroll
    // makes the most sense if we find we are having performance issues.
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        T residual_buffer[T_per_load];
        T bias_buffer[T_per_load];

        mem_access::load_global<ln::granularity>(
            iteration_buffer, input_base + i * stride, thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(residual_buffer,
                                                 residual_base + i * stride,
                                                 thread_offset + i * stride < elems_per_row);
        mem_access::load_global<ln::granularity>(
            bias_buffer, bias_base + i * stride, thread_offset + i * stride < elems_per_row);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            float vals_up_cast = conversion::to<float>(iteration_buffer[j]);
            float res_up_cast = conversion::to<float>(residual_buffer[j]);
            float bias_up_cast = conversion::to<float>(bias_buffer[j]);
            vals_up_cast += res_up_cast + bias_up_cast;
            partial_sum += vals_up_cast;
            iteration_buffer[j] = conversion::to<T>(vals_up_cast);
        }
    }

    const float sum = ln_sum_reduce<ln::max_warps>(tb, warp, sum_buffer, partial_sum);
    const float mean = sum / elems_per_row;

    float partial_mean_diff = 0.f;
#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            // Using a 0 value here skews the variance, have to if-guard
            if (thread_offset + i * stride < elems_per_row) {
                float diff = (conversion::to<float>(local_buffer[i * T_per_load + j]) - mean);
                partial_mean_diff += diff * diff;
            }
        }
    }

    const float mean_diff = ln_sum_reduce<ln::max_warps>(tb, warp, var_buffer, partial_mean_diff);
    const float variance = mean_diff / elems_per_row;
    const float denom = __frsqrt_rn(variance + epsilon);

    const T mean_compute = conversion::to<T>(mean);
    const T denom_compute = conversion::to<T>(denom);

    T* block_output = output + block_offset;

#pragma unroll
    for (int i = 0; i < UNROLL * ln::internal_unroll; i++) {
        T* iteration_buffer = local_buffer + i * T_per_load;
        const int iter_idx = i * stride + thread_offset;
        const bool do_loads = iter_idx < elems_per_row;

        T gamma_local[T_per_load], beta_local[T_per_load];

        mem_access::load_global<ln::granularity>(gamma_local, gamma + iter_idx, do_loads);
        mem_access::load_global<ln::granularity>(beta_local, beta + iter_idx, do_loads);

#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = (iteration_buffer[j] - mean_compute) * denom_compute;
            iteration_buffer[j] = iteration_buffer[j] * gamma_local[j] + beta_local[j];
        }

        if (do_loads) {
            mem_access::store_global<ln::granularity>(block_output + iter_idx, iteration_buffer);
        }
    }
    device_quantize<UNROLL, q_bits>(
        local_buffer, scales, out_int8, base_offset, thread_offset, stride, elems_per_row);
}

#define LAUNCH_FUSED_RES_LN_QUANT(T, unroll_factor, q_bits)                           \
    fused_residual_ln_quant<T, unroll_factor, q_bits><<<grid, block, 0, stream>>>( \
        output, out_int8, scales, vals, residual, bias, gamma, beta, epsilon, elems_per_row);

template <typename T, int q_bits>
void launch_fused_residual_ln_quant_impl(int8_t* out_int8,
                              T* output,
                              float* scales,
                              const T* vals,
                              const T* residual,
                              const T* bias,
                              const T* gamma,
                              const T* beta,
                              float epsilon,
                              int rows,
                              int elems_per_row,
                              cudaStream_t stream)
{
    constexpr int max_unroll = 4;

    // 8 for __half, 4 for float
    constexpr int T_per_load = ln::granularity / sizeof(T);
    // 32 for __half, 16 for float
    constexpr int T_per_thread_unroll = T_per_load * ln::internal_unroll;
    // 1024 for __half, 512 for float
    constexpr int T_per_warp_unroll = T_per_thread_unroll * hw_warp_size;

    int32_t unroll = 1;
    while (T_per_warp_unroll * ln::max_warps * unroll < elems_per_row) { unroll++; }

    const int warps =
        (elems_per_row + unroll * T_per_warp_unroll - 1) / (unroll * T_per_warp_unroll);

    dim3 grid(rows);
    dim3 block(warps * hw_warp_size);

    // This should match the max_unroll constexpr
    if (unroll == 1) {
        LAUNCH_FUSED_RES_LN_QUANT(T, 1, q_bits);
    } else if (unroll == 2) {
        LAUNCH_FUSED_RES_LN_QUANT(T, 2, q_bits);
    } else if (unroll == 3) {
        LAUNCH_FUSED_RES_LN_QUANT(T, 3, q_bits);
    } else if (unroll == 4) {
        LAUNCH_FUSED_RES_LN_QUANT(T, 4, q_bits);
    }
}

void launch_fused_residual_ln_quant(int8_t * out_int8,
                                    __half * output,
                                    float* scales,
                                    const __half* vals,
                                    const __half* residual,
                                    const __half* bias,
                                    const __half* gamma,
                                    const __half* beta,
                                    float epsilon,
                                    int rows,
                                    int elems_per_row,
                                    cudaStream_t stream) {
    launch_fused_residual_ln_quant_impl<__half, 8>(out_int8, output, scales, vals, residual, bias, gamma, beta, epsilon, rows, elems_per_row, stream);
}

void launch_fused_residual_ln_quant_int4(int8_t * out_int8,
                                    __half * output,
                                    float* scales,
                                    const __half* vals,
                                    const __half* residual,
                                    const __half* bias,
                                    const __half* gamma,
                                    const __half* beta,
                                    float epsilon,
                                    int rows,
                                    int elems_per_row,
                                    cudaStream_t stream){
    launch_fused_residual_ln_quant_impl<__half, 4>(out_int8, output, scales, vals, residual, bias, gamma, beta, epsilon, rows, elems_per_row, stream);
}