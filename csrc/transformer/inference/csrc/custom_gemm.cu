

#include <limits>
#include "custom_cuda_layers.h"

#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
namespace cg = cooperative_groups;

#define INPUT_TILE 1
#define INPUT_TILE1 1

// Input tile used in the gemm kernel v2
#define INPUT_TILE2_Q 5

#define INPUT_TILE2 10

#define MAX_REG_SIZE 20

#define WARP_SIZE 32
#define MAX_WARP_NUM 32
#define MAX_BLOCK_SUM 8

#define loop_unroll 4
#define loop_unroll_bits 2

#define inner_loop_unroll 4
#define inner_loop_unroll_bits 2

#define INT8WIDTH 2

#define MAX_QUANTIZE_GROUPING 1024

#define ACC_HALF true

inline __device__ float gelu(const float x)
{
    float y = 0.5 * x * (1.0 + tanhf(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
    return y;
}

#define INPUT_TILE11 10

__global__ void input_tiled_gemm_kernel_v2_fff(__half* output,
                                               const __half* vals,
                                               const __half* weight,
                                               const __half* bias,
                                               __half* block_sums,
                                               unsigned int hidden_dim,
                                               unsigned int block_reduce,
                                               unsigned int input_size,
                                               unsigned int output_size,
                                               unsigned int outputBlocks,
                                               unsigned int blockStride,
                                               bool add_gelu = false)
{
#if __CUDA_ARCH__ >= 700
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;

    int warp_num = blockDim.x >> 5;

    extern __shared__ float base1[];
    float4* shared_quantize_scale = (float4*)base1;
    __half2* shared_sum = (__half2*)&shared_quantize_scale[MAX_QUANTIZE_GROUPING >> 1] + 2112 +
                          (gid * (WARP_SIZE + 1)) + lane;

    // for (int j = 0; j < input_size; j++)
    {
        const float4* vals_cast = reinterpret_cast<const float4*>(vals);
        const float4* weight_cast = reinterpret_cast<const float4*>(weight);

        weight_cast += ((unsigned)(blockIdx.x / outputBlocks) * blockStride);
        vals_cast += (unsigned)(blockIdx.x / outputBlocks) * (hidden_dim >> 3);
        unsigned hidden_oct = (hidden_dim >> 3);
        weight_cast += (gid << 1) * output_size + (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
        int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
        float4 weight_q[2];
        if (col < output_size) {
            weight_q[0] = weight_cast[0];
            weight_q[1] = weight_cast[output_size];
        }
        float4* vals_h_shared =
            (float4*)(&shared_quantize_scale[MAX_QUANTIZE_GROUPING >> 1]) + (gid << 4);
        {
            // we read (loop_unroll >> 2) half-2 values per lane, and for 2 times of the
            // INPUT_TILE this makes more threads engaged in reading data from shared memory
            // into registers!
            if (lane < (INPUT_TILE11) && (lane) < input_size && gid < hidden_oct) {
                vals_h_shared[lane] = vals_cast[lane * (hidden_oct << block_reduce) + gid];
            }
            g.sync();
        }
        weight_cast += (output_size * (warp_num << 1));
        int iterations = hidden_dim / (warp_num << 3) - 2;

        {
            if (col < output_size) {
                float4 w_q[2];
                w_q[0] = weight_q[0];
                w_q[1] = weight_q[1];

                weight_q[0] = weight_cast[0];
                weight_q[1] = weight_cast[output_size];

                __half2* weight_h = (__half2*)(w_q);
                for (int t = 0; t < INPUT_TILE11; t++) {
                    __half* input_sh = (__half*)(vals_h_shared + t);
                    __half2 sum;
                    __half2 inp = __halves2half2(input_sh[0], input_sh[0]);

                    sum = inp * weight_h[0];
#pragma unroll
                    for (int li = 1; li < 8; li++) {
                        __half2 inp = __halves2half2(input_sh[li], input_sh[li]);
                        sum += inp * weight_h[li];
                    }
                    shared_sum[(t << 11)] = sum;
                }
            }
            vals_cast += warp_num;
            weight_cast += (output_size * (warp_num << 1));

            if (lane < (INPUT_TILE11) && (lane) < input_size && gid < hidden_oct) {
                vals_h_shared[lane] = vals_cast[lane * (hidden_oct << block_reduce) + gid];
            }
            g.sync();
        }
        for (int u = 0; u < iterations; u++) {
            if (col < output_size) {
                float4 w_q[2];
                w_q[0] = weight_q[0];
                w_q[1] = weight_q[1];

                weight_q[0] = weight_cast[0];
                weight_q[1] = weight_cast[output_size];

                __half2* weight_h = (__half2*)(w_q);
                for (int t = 0; t < INPUT_TILE11; t++) {
                    __half2 sum = (shared_sum[(t << 11)]);
                    __half* inp_data = (__half*)(vals_h_shared + t);
#pragma unroll
                    for (int li = 0; li < 8; li++) {
                        __half2 inp = __halves2half2(inp_data[li], inp_data[li]);
                        sum += inp * weight_h[li];
                    }
                    shared_sum[(t << 11)] = sum;
                }
            }
            vals_cast += warp_num;
            weight_cast += (output_size * (warp_num << 1));

            if (lane < (INPUT_TILE11) && (lane) < input_size && gid < hidden_oct) {
                vals_h_shared[lane] = vals_cast[lane * (hidden_oct << block_reduce) + gid];
            }
            g.sync();
        }
        __half2* weight_h = (__half2*)(weight_q);

        for (int t = 0; t < INPUT_TILE11; t++) {
            __half2 sum = (shared_sum[(t << 11)]);
            __half* inp_data = (__half*)(vals_h_shared + t);
#pragma unroll
            for (int li = 0; li < 8; li++) {
                __half2 inp = __halves2half2(inp_data[li], inp_data[li]);
                sum += inp * weight_h[li];
            }
            shared_sum[(t << 11)] = sum;
        }

        {
            int col = (blockIdx.x % outputBlocks) * WARP_SIZE + lane;
            const __half2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);

            __half2* output_cast =
                reinterpret_cast<__half2*>(((gridDim.x == outputBlocks) ? output : block_sums));
            output_cast += (unsigned)(blockIdx.x / outputBlocks) * (output_size);
            __half2* partial_result = (__half2*)&shared_quantize_scale[MAX_QUANTIZE_GROUPING >> 1] +
                                      2112 + lane * (WARP_SIZE + 1) + gid;

            __syncthreads();

            for (int t = 0; t < INPUT_TILE11; t++) {
                __half2 sum_f[2];
                sum_f[0] = (partial_result[t << 11]);
#pragma unroll
                for (int i = 1; i < WARP_SIZE; i *= 2) {
                    sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
                    sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
                }

                if (lane == 0) { partial_result[gid] = sum_f[0]; }
            }

            __syncthreads();
            if (gid < INPUT_TILE11 && gid < input_size) {
                if (col < output_size) {
                    __half2 sum_f[1];
                    sum_f[0] = partial_result[lane];
                    if (bias && blockIdx.x < outputBlocks) {
                        __half2 bias_ff = bias_cast[col];
                        sum_f[0].x += bias_ff.x;
                        sum_f[0].y += bias_ff.y;
                        if (add_gelu && gridDim.x == outputBlocks) {
                            sum_f[0].x = gelu(sum_f[0].x);
                            sum_f[0].y = gelu(sum_f[0].y);
                        }
                    }
                    output_cast[col + (gid * (output_size << block_reduce))] = sum_f[0];
                }
            }
        }
    }
#endif
}

__global__ void block_reduce_kernel(float* output,
                                    float* block_sums,
                                    int batch,
                                    int output_size,
                                    bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    unsigned total_count = batch * output_size;

    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;
    unsigned int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    float2* block_sums_cast = reinterpret_cast<float2*>(block_sums);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    block_sums_cast += gid * output_size;

    if (col_index < total_count) {
        __shared__ float2 data_shared[MAX_WARP_NUM * (WARP_SIZE + 1)];

        data_shared[gid * (WARP_SIZE) + lane] =
            block_sums_cast[(col_index / output_size) * (warp_num * output_size) +
                            col_index % output_size];

        b.sync();

        float2 data = data_shared[(lane % warp_num) * WARP_SIZE + gid * (WARP_SIZE / warp_num) +
                                  (lane / warp_num)];

        b.sync();
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            data.x += g.shfl_down(data.x, i);
            data.y += g.shfl_down(data.y, i);
        }

        if ((lane % warp_num) == 0) {
            if (add_gelu) {
                data.x = gelu(data.x);
                data.y = gelu(data.y);
            }
            data_shared[gid * (WARP_SIZE / warp_num) + (lane / warp_num)] = (data);
        }

        b.sync();

        if (gid == 0) output_cast[col_index] = data_shared[lane];
    }
}
__global__ void block_reduce_kernel(__half* output,
                                    __half* block_sums,
                                    unsigned batch,
                                    unsigned int output_size,
                                    bool add_gelu = false)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);
    unsigned total_count = batch * output_size;
    unsigned int gid = threadIdx.x >> 5;
    unsigned int lane = threadIdx.x & 0x1f;
    unsigned int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    __half2* block_sums_cast = reinterpret_cast<__half2*>(block_sums);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    block_sums_cast += gid * output_size;

    if (col_index < total_count) {
        __shared__ __half2 data_shared[MAX_WARP_NUM * (WARP_SIZE + 1)];

        data_shared[gid * (WARP_SIZE) + lane] =
            block_sums_cast[(col_index / output_size) * (warp_num * output_size) +
                            col_index % output_size];

        b.sync();

        float2 data = __half22float2(data_shared[(lane % warp_num) * WARP_SIZE +
                                                 gid * (WARP_SIZE / warp_num) + (lane / warp_num)]);

        b.sync();
#pragma unroll
        for (int i = 1; i < warp_num; i <<= 1) {
            data.x += g.shfl_xor(data.x, i);
            data.y += g.shfl_xor(data.y, i);
        }

        if ((lane % warp_num) == 0) {
            if (add_gelu) {
                data.x = gelu(data.x);
                data.y = gelu(data.y);
            }
            data_shared[gid * (WARP_SIZE / warp_num) + (lane / warp_num)] = __float22half2_rn(data);
        }

        b.sync();

        if (gid == 0) output_cast[col_index] = data_shared[lane];
    }
}

template <>
void launch_input_tiled_gemm_kernel_v2<float>(float* output,
                                              const float* vals,
                                              const float* weight,
                                              const float* bias,
                                              float* block_sums,
                                              unsigned int hidden_dim,
                                              unsigned int input_size,
                                              unsigned int output_size,
                                              bool add_gelu,
                                              cudaStream_t stream)
{
}

template <>
void launch_input_tiled_gemm_kernel_v2<__half>(__half* output,
                                               const __half* vals,
                                               const __half* weight,
                                               const __half* bias,
                                               __half* block_sums,
                                               unsigned int hidden_dim,
                                               unsigned int input_size,
                                               unsigned int output_size,
                                               bool add_gelu,
                                               cudaStream_t stream)
{
    output_size /= 2;
    int outputBlocks = (output_size - 1) / WARP_SIZE + 1;

    int block_reduce = (SMs > outputBlocks ? SMs / outputBlocks : 1);
    int br2 = (int)log2(block_reduce);
    block_reduce = (int)pow(2.0, (float)br2);

    int threads = ((hidden_dim / block_reduce) >> 1);
    if (threads > 1024) threads = 1024;
    int blockStride = ((output_size >> 2) * hidden_dim) / block_reduce;
    // printf("block_reduce =  %d , br2 = %d ,  threads = %d\n, hidden: %d, output: %d",
    // block_reduce, br2, threads, hidden_dim, output_size);
    dim3 grid_dim(outputBlocks * block_reduce);
    dim3 block_dim(threads);
    // printf("hidden=%d, out_size:%d, \t blocks: %d, threads: %d \n", hidden_dim, output_size,
    // grid_dim.x, block_dim.x);
    cudaFuncSetAttribute(
        input_tiled_gemm_kernel_v2_fff, cudaFuncAttributeMaxDynamicSharedMemorySize, 98160);
    input_tiled_gemm_kernel_v2_fff<<<grid_dim, block_dim, 98160, stream>>>(
        output,
        vals,
        weight,
        bias,
        block_sums,
        hidden_dim / block_reduce,
        br2,
        input_size,
        output_size,
        outputBlocks,
        blockStride,
        add_gelu);
    if (block_reduce > 1) {
        dim3 grids(((output_size * input_size) - 1) / WARP_SIZE + 1);
        dim3 blocks(block_reduce * WARP_SIZE);
        block_reduce_kernel<<<grids, blocks, 0, stream>>>(
            output, block_sums, input_size, (output_size), add_gelu);
    }
}

__global__ void input_tiled_gemm_kernel_gelu(__half* output,
                                             __half* residual_add,
                                             const __half* vals,
                                             const __half* residual,
                                             const __half* input_bias,
                                             const __half* weight,
                                             const __half* bias,
                                             const __half* gamma,
                                             const __half* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             bool preLN)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    const __half2* residual_cast = reinterpret_cast<const __half2*>(residual);
    __half2* residual_add_cast = reinterpret_cast<__half2*>(residual_add);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);
    const __half2* input_bias_cast = reinterpret_cast<const __half2*>(input_bias);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ __half2 input_shared[9000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                __half2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    float2 inp_f = __half22float2(inp_reg[k]);
                    float2 residual_f =
                        __half22float2(residual_cast[(j + t) * hidden_half + input_id]);
                    float2 bias_f = __half22float2(input_bias_cast[input_id]);
                    inp_f.x += residual_f.x + bias_f.x;
                    inp_f.y += residual_f.y + bias_f.y;
                    inp_reg[k] = __float22half2_rn(inp_f);
                    // if (preLN) residual_add_cast[(j + t) * hidden_half + input_id] = inp_reg[k];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                // b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    sum += inp_f.x + inp_f.y;
                }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    float2 inp_f = __half22float2(inp_reg[f]);
                    inp_f.x -= mean;
                    inp_f.y -= mean;
                    inp_reg[f] = __float22half2_rn(inp_f);
                    sum += inp_f.x * inp_f.x;
                    sum += inp_f.y * inp_f.y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                __half2 variance_h = __float2half2_rn(sum);
                const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
                const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
                for (int f = 0; f < k; f++) {
                    int tid = f * blockDim.x + threadIdx.x;
                    inp_reg[f] = inp_reg[f] * variance_h;
                    inp_reg[f] = inp_reg[f] * gamma_cast[tid] + beta_cast[tid];
                    input_shared[tid + t * hidden_half] = inp_reg[f];
                    if (!preLN) residual_add_cast[(j + t) * hidden_half + tid] = inp_reg[f];
                    // output_cast[(j + t) * hidden_half + tid] = inp_reg[f];
                }
                b.sync();
            }
        }

        int wid = gid << 2;
        int offset = wid * output_size;
        float2 sum[INPUT_TILE];
        for (int t = 0; t < INPUT_TILE; t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        while (wid < hidden_dim) {
            __half2 vals_f[INPUT_TILE * 4];
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    __half2 val_h[2];
                    val_h[0] = input_shared[t * hidden_half + (wid >> 1)];
                    val_h[1] = input_shared[t * hidden_half + (wid >> 1) + 1];

                    __half* inp_data[2];
                    inp_data[0] = reinterpret_cast<__half*>(&val_h[0]);
                    inp_data[1] = reinterpret_cast<__half*>(&val_h[1]);

                    vals_f[(t << 2)] = __halves2half2(inp_data[0][0], inp_data[0][0]);
                    vals_f[(t << 2) + 1] = __halves2half2(inp_data[0][1], inp_data[0][1]);
                    vals_f[(t << 2) + 2] = __halves2half2(inp_data[1][0], inp_data[1][0]);
                    vals_f[(t << 2) + 3] = __halves2half2(inp_data[1][1], inp_data[1][1]);
                }
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                __half2 weight_h[4];
                weight_h[0] = weight_cast[offset1];
                weight_h[1] = weight_cast[output_size + offset1];
                weight_h[2] = weight_cast[(output_size << 1) + offset1];
                weight_h[3] = weight_cast[((output_size << 1) + output_size) + offset1];
#pragma unroll
                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        float2 mul[4];
                        mul[0] = __half22float2(vals_f[(t << 2)] * weight_h[0]);
                        mul[1] = __half22float2(vals_f[(t << 2) + 1] * weight_h[1]);
                        mul[2] = __half22float2(vals_f[(t << 2) + 2] * weight_h[2]);
                        mul[3] = __half22float2(vals_f[(t << 2) + 3] * weight_h[3]);

                        sum[t].x += mul[0].x + mul[1].x + mul[2].x + mul[3].x;
                        sum[t].y += mul[0].y + mul[1].y + mul[2].y + mul[3].y;
                    }
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 4;
            offset += (output_size * warp_num * 4);
        }
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 sum_g = sum[t];
                __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
                const __half2* bias_cast = reinterpret_cast<const __half2*>(bias);
                {
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    //__syncthreads();
                }

#pragma unroll
                for (int i = 1; i < WARP_SIZE; i *= 2) {
                    sum_g.x += g.shfl_xor(sum_g.x, i);
                    sum_g.y += g.shfl_xor(sum_g.y, i);
                }

                if (lane == 0) { partial_result[0][gid] = sum_g; }
                __syncthreads();

                if (gid == 0) {
                    int col = blockIdx.x * WARP_SIZE + lane;
                    if (col < output_size) {
                        sum_g = partial_result[0][lane];
                        float2 bias_f = __half22float2(bias_cast[col]);
                        sum_g.x = bias_f.x + sum_g.x;
                        sum_g.y = bias_f.y + sum_g.y;
                        sum_g.x = gelu(sum_g.x);
                        sum_g.y = gelu(sum_g.y);

                        output_cast[(j + t) * output_size + col] = __float22half2_rn(sum_g);
                    }
                }
            }
        }
    }
#endif
}

__global__ void input_tiled_gemm_kernel_gelu(float* output,
                                             float* residual_add,
                                             const float* vals,
                                             const float* residual,
                                             const float* input_bias,
                                             const float* weight,
                                             const float* bias,
                                             const float* gamma,
                                             const float* beta,
                                             const float epsilon,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             bool preLN)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;
    int id = threadIdx.x;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* residual_cast = reinterpret_cast<const float2*>(residual);
    float2* residual_add_cast = reinterpret_cast<float2*>(residual_add);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);
    const float2* input_bias_cast = reinterpret_cast<const float2*>(input_bias);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += INPUT_TILE) {
        __shared__ float2 input_shared[5000];
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 inp_reg[8];
                int k = 0;
                int input_id = id;
                while (input_id < hidden_half) {
                    inp_reg[k] = vals_cast[(j + t) * hidden_half + input_id];
                    float2 residual_f = residual_cast[(j + t) * hidden_half + input_id];
                    float2 bias_f = input_bias_cast[input_id];
                    inp_reg[k].x += residual_f.x + bias_f.x;
                    inp_reg[k].y += residual_f.y + bias_f.y;
                    if (preLN) residual_add_cast[(j + t) * hidden_half + input_id] = inp_reg[k];
                    input_shared[input_id + t * hidden_half] = inp_reg[k++];
                    input_id += blockDim.x;
                }
                b.sync();

                float sum = 0;
                for (int f = k - 1; f >= 0; f--) { sum += inp_reg[f].x + inp_reg[f].y; }
                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                __shared__ float shr[MAX_WARP_NUM];
                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                float mean = sum / hidden_dim;
                sum = 0.f;
                for (int f = 0; f < k; f++) {
                    inp_reg[f].x -= mean;
                    inp_reg[f].y -= mean;
                    sum += inp_reg[f].x * inp_reg[f].x;
                    sum += inp_reg[f].y * inp_reg[f].y;
                }

                for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }

                if (g.thread_rank() == 0) shr[gid] = sum;
                b.sync();
                if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
                for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }

                sum = g.shfl(sum, 0);
                sum /= hidden_dim;
                sum += epsilon;
                sum = __frsqrt_rn(sum);
                const float2* gamma_cast = reinterpret_cast<const float2*>(gamma);
                const float2* beta_cast = reinterpret_cast<const float2*>(beta);
                for (int f = 0; f < k; f++) {
                    int id = f * blockDim.x + threadIdx.x;
                    inp_reg[f].x = inp_reg[f].x * sum;
                    inp_reg[f].y = inp_reg[f].y * sum;

                    inp_reg[f].x = inp_reg[f].x * gamma_cast[id].x + beta_cast[id].x;
                    inp_reg[f].y = inp_reg[f].y * gamma_cast[id].y + beta_cast[id].y;

                    if (!preLN) residual_add_cast[(j + t) * hidden_half + id] = inp_reg[f];
                    input_shared[id + t * hidden_half] = inp_reg[f];
                }
                b.sync();
            }
        }

        int wid = gid << 1;
        int offset = wid * output_size;
        float2 sum[INPUT_TILE];
        for (int t = 0; t < INPUT_TILE; t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        while (wid < hidden_dim) {
            float2 val_data[INPUT_TILE];
            for (int t = 0; t < INPUT_TILE; t++) {
                if ((t + j) < input_size) {
                    val_data[t] = input_shared[t * hidden_half + (wid >> 1)];
                }
            }

            int row = blockIdx.x * WARP_SIZE + lane;
            auto offset1 = offset + row;
            while (row < output_size) {
                float2 weight[2];
                weight[0] = weight_cast[offset1];
                weight[1] = weight_cast[output_size + offset1];

                for (int t = 0; t < INPUT_TILE; t++) {
                    if ((t + j) < input_size) {
                        float2 mul[2];
                        mul[0].x = val_data[t].x * weight[0].x;
                        mul[0].y = val_data[t].x * weight[0].y;
                        mul[1].x = val_data[t].y * weight[1].x;
                        mul[1].y = val_data[t].y * weight[1].y;

                        sum[t].x += mul[0].x + mul[1].x;
                        sum[t].y += mul[0].y + mul[1].y;
                    }
                }
                row += (gridDim.x * WARP_SIZE);
                offset1 += (gridDim.x * WARP_SIZE);
            }
            wid += warp_num * 2;
            offset += (output_size * warp_num * 2);
        }
        for (int t = 0; t < INPUT_TILE; t++) {
            if ((t + j) < input_size) {
                float2 sum_g = sum[t];
                __shared__ float2 partial_result[MAX_WARP_NUM][WARP_SIZE + 1];
                const float2* bias_cast = reinterpret_cast<const float2*>(bias);
                {
                    partial_result[gid][lane] = sum_g;
                    __syncthreads();
                    sum_g = partial_result[lane][gid];
                    __syncthreads();
                }

#pragma unroll
                for (int i = 1; i < WARP_SIZE; i *= 2) {
                    sum_g.x += g.shfl_xor(sum_g.x, i);
                    sum_g.y += g.shfl_xor(sum_g.y, i);
                }

                if (lane == 0) { partial_result[0][gid] = sum_g; }
                __syncthreads();

                if (gid == 0) {
                    int col = blockIdx.x * WARP_SIZE + lane;
                    if (col < output_size) {
                        sum_g = partial_result[0][lane];
                        float2 bias_f = bias_cast[col];
                        sum_g.x = bias_f.x + sum_g.x;
                        sum_g.y = bias_f.y + sum_g.y;
                        sum_g.x = gelu(sum_g.x);
                        sum_g.y = gelu(sum_g.y);

                        output_cast[(j + t) * output_size + col] = sum_g;
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel_gelu(T* output,
                                         T* residual_add,
                                         const T* vals,
                                         const T* residual,
                                         const T* input_bias,
                                         const T* weight,
                                         const T* bias,
                                         const T* gamma,
                                         const T* beta,
                                         const float epsilon,
                                         int hidden_dim,
                                         int input_size,
                                         int output_size,
                                         bool preLN,
                                         cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel_gelu<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                     residual_add,
                                                                     vals,
                                                                     residual,
                                                                     input_bias,
                                                                     weight,
                                                                     bias,
                                                                     gamma,
                                                                     beta,
                                                                     epsilon,
                                                                     hidden_dim,
                                                                     input_size,
                                                                     output_size,
                                                                     preLN);
}

template void launch_input_tiled_gemm_kernel_gelu(float* output,
                                                  float* residual_add,
                                                  const float* vals,
                                                  const float* residual,
                                                  const float* input_bias,
                                                  const float* weight,
                                                  const float* bias,
                                                  const float* gamma,
                                                  const float* beta,
                                                  const float epsilon,
                                                  int hidden_dim,
                                                  int input_size,
                                                  int output_size,
                                                  bool preLN,
                                                  cudaStream_t stream);

template void launch_input_tiled_gemm_kernel_gelu(__half* output,
                                                  __half* residual_add,
                                                  const __half* vals,
                                                  const __half* residual,
                                                  const __half* input_bias,
                                                  const __half* weight,
                                                  const __half* bias,
                                                  const __half* gamma,
                                                  const __half* beta,
                                                  const float epsilon,
                                                  int hidden_dim,
                                                  int input_size,
                                                  int output_size,
                                                  bool preLN,
                                                  cudaStream_t stream);

/*
__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const __half* weight,
                                        const __half* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    __half2 ZERO_h2 = __float2half2_rn(0.f);
    {
        float2 sum;
        __half2 *sum_h = (__half2*)(&sum);
        sum_h[0] = __float2half2_rn(0.f);
        sum_h[1] = __float2half2_rn(0.f);

        {
            int wid = gid << loop_unroll_bits;
            weight_cast += (wid * output_size + col_index);

            while (wid < hidden_dim)
            {
                __half2 vals_f[loop_unroll * (INPUT_TILE1)];
                {
                    {
                        {
                            float2 val_h;
                            val_h = vals_cast[gid];

                            __half* inp_data;
                            inp_data = reinterpret_cast<__half*>(&val_h);

                            vals_f[0] = __halves2half2(inp_data[0], inp_data[0]);
                            vals_f[1] = __halves2half2(inp_data[1], inp_data[1]);
                            vals_f[2] = __halves2half2(inp_data[2], inp_data[2]);
                            vals_f[3] = __halves2half2(inp_data[3], inp_data[3]);
                        }
                    }
                }

                    float2 weight_f[loop_unroll];
                    __half2 *weight_h = (__half2*)weight_f;
#pragma unroll
                    for (int k = 0; k < loop_unroll; k++)
                        weight_f[k] = weight_cast[k * output_size];

#pragma unroll
                    for (int k = 0; k < (loop_unroll >> inner_loop_unroll_bits); k++)
#pragma unroll
                                for (int li = 0; li < inner_loop_unroll; li++) {
                                    weight_h[0] = (vals_f[li] * weight_h[li<<1]);
                                    weight_h[1] = (vals_f[li] * weight_h[(li<<1)+1]);
                                    sum_h[0] += weight_h[0];
                                    sum_h[1] += weight_h[1];
                                }
                wid += (warp_num << loop_unroll_bits);
                weight_cast += (output_size * (warp_num << loop_unroll_bits));
            }
        }
        {
            const float2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
            __shared__ float2 partial_result[2 * MAX_WARP_NUM * (WARP_SIZE + 1)];

            {

              {
                    __half2 *sum_g_h = (__half2*)(&sum);
                    float2 sum_f[2];
                    sum_f[0] = __half22float2(sum_g_h[0]);
                    sum_f[1] = __half22float2(sum_g_h[1]);
                    partial_result[(gid << 1) * (WARP_SIZE + 1) + lane] = sum_f[0];
                    partial_result[((gid << 1) + 1) * (WARP_SIZE + 1) + lane] = sum_f[1];

                    b.sync();
                    sum_f[0] = partial_result[(lane << 1) * (WARP_SIZE + 1) + gid];
                    sum_f[1] = partial_result[((lane << 1) + 1) * (WARP_SIZE + 1) + gid];

                    b.sync();
#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_f[0].x += g.shfl_xor(sum_f[0].x, i);
                        sum_f[0].y += g.shfl_xor(sum_f[0].y, i);
                        sum_f[1].x += g.shfl_xor(sum_f[1].x, i);
                        sum_f[1].y += g.shfl_xor(sum_f[1].y, i);
                    }

                    if (lane <= 1) { partial_result[(gid << 1) + lane] = sum_f[lane]; }

                    b.sync();

                    if (gid == 0) {
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            sum_f[0] = partial_result[(lane << 1)];
                            sum_f[1] = partial_result[((lane << 1) + 1)];
                            if (bias) {
                                float2 bias_f = bias_cast[col];
                                __half2 *bias_h = (__half2*)(&bias_f);
                                sum_f[0].x += (float)bias_h[0].x;
                                sum_f[0].y += (float)bias_h[0].y;
                                sum_f[1].x += (float)bias_h[1].x;
                                sum_f[1].y += (float)bias_h[1].y;
                            }
                            sum_g_h[0] = __float22half2_rn(sum_f[0]);
                            sum_g_h[1] = __float22half2_rn(sum_f[1]);
                            output_cast[col + output_size] = sum;
                        }
                    }
                }
            }
        }
        weight_cast = reinterpret_cast<const float2*>(weight);
    }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output,
                                        const float* vals,
                                        const float* weight,
                                        const float* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
        float2 sum[INPUT_TILE1];
#pragma unroll
        for (int t = 0; t < (INPUT_TILE1); t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        {
            int wid = gid << 1;
            int offset = wid * output_size;

            while (wid < hidden_dim) {
                float2 val_data[INPUT_TILE1];
                {
                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            val_data[t] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
                        }
                    }
                }

                int row = blockIdx.x * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    float2 weight[2];
                    weight[0] = weight_cast[offset1];
                    weight[1] = weight_cast[output_size + offset1];

                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[2];
                            mul[0].x = val_data[t].x * weight[0].x;
                            mul[0].y = val_data[t].x * weight[0].y;
                            mul[1].x = val_data[t].y * weight[1].x;
                            mul[1].y = val_data[t].y * weight[1].y;

                            sum[t].x += mul[0].x + mul[1].x;
                            sum[t].y += mul[0].y + mul[1].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 2;
                offset += (output_size * warp_num * 2);
            }
        }
        {
            const float2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
            __shared__ float2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < (INPUT_TILE1); t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid * (WARP_SIZE + 1) + lane] = sum_g;
                    __syncthreads();

                    sum_g = partial_result[lane * (WARP_SIZE + 1) + gid];
                    __syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[gid] = sum_g; }

                    __syncthreads();

                    if (gid == 0) {
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            sum_g = partial_result[lane];
                            if (bias) {
                                float2 bias_f = bias_cast[col];
                                sum_g.x += bias_f.x;
                                sum_g.y += bias_f.y;
                            }
                            output_cast[col + (j + t) * output_size] = sum_g;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 4;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    //printf("threads: %d, blocks: %d \n", threads, (output_size - 1) / WARP_SIZE + 1);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, weight, bias, hidden_dim, input_size, output_size);
}


template void launch_input_tiled_gemm_kernel(float* output,
                                             const float* vals,
                                             const float* weight,
                                             const float* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const __half* weight,
                                             const __half* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream);
*/

__global__ void input_tiled_gemm_kernel(__half* output,
                                        const __half* vals,
                                        const __half* weight,
                                        const __half* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        bool add_gelu)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;

    __half2 sum;
    sum = __float2half2_rn(0.f);

    {
        int wid = gid << loop_unroll_bits;
        weight_cast += (wid * output_size + col_index);

        while (wid < hidden_dim) {
            __half2 vals_f[loop_unroll * (INPUT_TILE1)];
            {
                float2 val_h;
                val_h = vals_cast[(wid >> 2)];

                __half* inp_data;
                inp_data = reinterpret_cast<__half*>(&val_h);

                vals_f[0] = __halves2half2(inp_data[0], inp_data[0]);
                vals_f[1] = __halves2half2(inp_data[1], inp_data[1]);
                vals_f[2] = __halves2half2(inp_data[2], inp_data[2]);
                vals_f[3] = __halves2half2(inp_data[3], inp_data[3]);
            }

            if (col_index < output_size) {
                __half2 weight_h[loop_unroll];
#pragma unroll
                for (int k = 0; k < loop_unroll; k++) weight_h[k] = weight_cast[k * output_size];

#pragma unroll
                for (int k = 0; k < (loop_unroll >> inner_loop_unroll_bits); k++)
#pragma unroll
                    for (int li = 0; li < inner_loop_unroll; li++) {
                        weight_h[0] = (vals_f[li] * weight_h[li]);
                        sum += weight_h[0];
                    }
            }
            wid += (warp_num << loop_unroll_bits);
            weight_cast += (output_size * (warp_num << loop_unroll_bits));
        }
    }
    {
        const __half2* bias_cast;
        if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
        __shared__ __half2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

        partial_result[gid * (WARP_SIZE + 1) + lane] = sum;

        b.sync();
        float2 sum_f;
        sum_f = __half22float2(partial_result[lane * (WARP_SIZE + 1) + gid]);

        b.sync();
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum_f.x += g.shfl_xor(sum_f.x, i);
            sum_f.y += g.shfl_xor(sum_f.y, i);
        }

        if (lane == 0) { partial_result[gid] = __float22half2_rn(sum_f); }

        b.sync();

        if (gid == 0) {
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) {
                sum = partial_result[lane];
                if (bias) {
                    float2 bias_f = __half22float2(bias_cast[col]);
                    sum_f = __half22float2(sum);
                    sum_f.x += bias_f.x;
                    sum_f.y += bias_f.y;
                    if (add_gelu) {
                        sum_f.x = gelu(sum_f.x);
                        sum_f.y = gelu(sum_f.y);
                    }
                    sum = __float22half2_rn(sum_f);
                }
                output_cast[col] = sum;
            }
        }
    }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output,
                                        const float* vals,
                                        const float* weight,
                                        const float* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        bool add_gelu)
{
    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    float2* output_cast = reinterpret_cast<float2*>(output);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const float2* weight_cast = reinterpret_cast<const float2*>(weight);

    int hidden_half = hidden_dim >> 1;

    for (int j = 0; j < input_size; j += (INPUT_TILE1)) {
        float2 sum[INPUT_TILE1];
#pragma unroll
        for (int t = 0; t < (INPUT_TILE1); t++) {
            sum[t].x = 0;
            sum[t].y = 0;
        }

        {
            int wid = gid << 1;
            int offset = wid * output_size;

            while (wid < hidden_dim) {
                float2 val_data[INPUT_TILE1];
                {
                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            val_data[t] = vals_cast[(j + t) * hidden_half + (wid >> 1)];
                        }
                    }
                }

                int row = blockIdx.x * WARP_SIZE + lane;
                auto offset1 = offset + row;
                while (row < output_size) {
                    float2 weight[2];
                    weight[0] = weight_cast[offset1];
                    weight[1] = weight_cast[output_size + offset1];

                    for (int t = 0; t < INPUT_TILE1; t++) {
                        if ((t + j) < input_size) {
                            float2 mul[2];
                            mul[0].x = val_data[t].x * weight[0].x;
                            mul[0].y = val_data[t].x * weight[0].y;
                            mul[1].x = val_data[t].y * weight[1].x;
                            mul[1].y = val_data[t].y * weight[1].y;

                            sum[t].x += mul[0].x + mul[1].x;
                            sum[t].y += mul[0].y + mul[1].y;
                        }
                    }
                    row += (gridDim.x * WARP_SIZE);
                    offset1 += (gridDim.x * WARP_SIZE);
                }
                wid += warp_num * 2;
                offset += (output_size * warp_num * 2);
            }
        }
        {
            const float2* bias_cast;
            if (bias) bias_cast = reinterpret_cast<const float2*>(bias);
            __shared__ float2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

            for (int t = 0; t < (INPUT_TILE1); t++) {
                if ((t + j) < input_size) {
                    float2 sum_g = sum[t];
                    partial_result[gid * (WARP_SIZE + 1) + lane] = sum_g;
                    __syncthreads();

                    sum_g = partial_result[lane * (WARP_SIZE + 1) + gid];
                    __syncthreads();

#pragma unroll
                    for (int i = 1; i < WARP_SIZE; i *= 2) {
                        sum_g.x += g.shfl_xor(sum_g.x, i);
                        sum_g.y += g.shfl_xor(sum_g.y, i);
                    }

                    if (lane == 0) { partial_result[gid] = sum_g; }

                    __syncthreads();

                    if (gid == 0) {
                        int col = blockIdx.x * WARP_SIZE + lane;
                        if (col < output_size) {
                            sum_g = partial_result[lane];
                            if (bias) {
                                float2 bias_f = bias_cast[col];
                                sum_g.x += bias_f.x;
                                sum_g.y += bias_f.y;
                            }
                            if (add_gelu) {
                                sum_g.x = gelu(sum_g.x);
                                sum_g.y = gelu(sum_g.y);
                            }
                            output_cast[col + (j + t) * output_size] = sum_g;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void launch_input_tiled_gemm_kernel(T* output,
                                    const T* vals,
                                    const T* weight,
                                    const T* bias,
                                    int hidden_dim,
                                    int input_size,
                                    int output_size,
                                    cudaStream_t stream,
                                    bool add_gelu)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, weight, bias, hidden_dim, input_size, output_size, add_gelu);
}

template void launch_input_tiled_gemm_kernel(float* output,
                                             const float* vals,
                                             const float* weight,
                                             const float* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream,
                                             bool add_gelu);

template void launch_input_tiled_gemm_kernel(__half* output,
                                             const __half* vals,
                                             const __half* weight,
                                             const __half* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             cudaStream_t stream,
                                             bool add_gelu);

#define NORM_REG 32

__device__ void layer_nrom_device(__half* output,
                                  const __half* vals,
                                  const __half* gamma,
                                  const __half* beta,
                                  float epsilon,
                                  int row_stride)
{
#if __CUDA_ARCH__ >= 700
    int iteration_stride = blockDim.x;
    int iterations = row_stride / iteration_stride;

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int id = threadIdx.x;
    int gid = id >> 5;
    int warp_num = iteration_stride >> 5;

    __half2 inp_reg[8];

    const __half2* vals_cast = reinterpret_cast<const __half2*>(vals);
    __half2* out_cast = reinterpret_cast<__half2*>(output);

    int k = 0;
    int input_id = id;
    while (input_id < row_stride) {
        inp_reg[k++] = vals_cast[input_id];
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
        out_cast[out_id] = inp_reg[f];
    }
    b.sync();
#endif
}

__global__ void input_tiled_gemm_kernel(__half* output,
                                        __half* intermediate,
                                        const __half* vals,
                                        const __half* weight,
                                        const __half* gamma,
                                        const __half* beta,
                                        float epsilon,
                                        const __half* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        int intm_size)
{
#if __CUDA_ARCH__ >= 700

    cg::thread_block b = cg::this_thread_block();
    cg::thread_block_tile<32> g = cg::tiled_partition<32>(b);

    int gid = threadIdx.x >> 5;
    int lane = threadIdx.x & 0x1f;
    int warp_num = blockDim.x >> 5;

    __half2* output_cast = reinterpret_cast<__half2*>(output);
    __half2* intm_cast = reinterpret_cast<__half2*>(intermediate);
    const float2* vals_cast = reinterpret_cast<const float2*>(vals);
    const __half2* weight_cast = reinterpret_cast<const __half2*>(weight);

    unsigned int col_index = blockIdx.x * WARP_SIZE + lane;
    __half2 sum;
    sum = __float2half2_rn(0.f);
    // layer_nrom_device((__half*)vals_cast, vals, gamma, beta, epsilon,hidden_dim/2);

    int id = threadIdx.x;

    {
        //__shared__ float2 vals_cast[4096];
        //__half2* input_shared = (__half2*)vals_cast;
        //{
        //    int hidden_half = hidden_dim >> 1;
        //    __half2 inp_reg[8];
        //    int k = 0;
        //    int input_id = id;
        //    while (input_id < hidden_half) {
        //        inp_reg[k] = vals_cast1[input_id];
        //        input_shared[input_id] = inp_reg[k++];
        //        input_id += blockDim.x;
        //    }
        //    b.sync();
        //    float sum = 0;
        //    for (int f = k - 1; f >= 0; f--) {
        //        float2 inp_f = __half22float2(inp_reg[f]);
        //        sum += inp_f.x + inp_f.y;
        //    }
        //    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }
        //    __shared__ float shr[MAX_WARP_NUM];
        //    if (g.thread_rank() == 0) shr[gid] = sum;
        //    b.sync();
        //    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        //    for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }
        //    sum = g.shfl(sum, 0);
        //    float mean = sum / hidden_dim;
        //    sum = 0.f;
        //    for (int f = 0; f < k; f++) {
        //        float2 inp_f = __half22float2(inp_reg[f]);
        //        inp_f.x -= mean;
        //        inp_f.y -= mean;
        //        inp_reg[f] = __float22half2_rn(inp_f);
        //        sum += inp_f.x * inp_f.x;
        //        sum += inp_f.y * inp_f.y;
        //    }
        //    for (int i = 1; i < 32; i *= 2) { sum += g.shfl_down(sum, i); }
        //    if (g.thread_rank() == 0) shr[gid] = sum;
        //    b.sync();
        //    if (g.thread_rank() < (warp_num)) sum = shr[g.thread_rank()];
        //    for (int i = 1; i < (warp_num); i *= 2) { sum += g.shfl_down(sum, i); }
        //    sum = g.shfl(sum, 0);
        //    sum /= hidden_dim;
        //    sum += epsilon;
        //    sum = __frsqrt_rn(sum);
        //    __half2 variance_h = __float2half2_rn(sum);
        //    const __half2* gamma_cast = reinterpret_cast<const __half2*>(gamma);
        //    const __half2* beta_cast = reinterpret_cast<const __half2*>(beta);
        //    for (int f = 0; f < k; f++) {
        //        int id = f * blockDim.x + threadIdx.x;
        //        inp_reg[f] = inp_reg[f] * variance_h;
        //        inp_reg[f] = inp_reg[f] * gamma_cast[id] + beta_cast[id];
        //        input_shared[id] = inp_reg[f];
        //    }
        //    b.sync();
        //}
        int wid = gid << loop_unroll_bits;
        weight_cast += (wid * output_size + col_index);
        while (wid < hidden_dim) {
            __half2 vals_f[loop_unroll * (INPUT_TILE1)];
            {
                float2 val_h;
                val_h = vals_cast[(wid >> 2)];
                __half* inp_data;
                inp_data = reinterpret_cast<__half*>(&val_h);
                vals_f[0] = __halves2half2(inp_data[0], inp_data[0]);
                vals_f[1] = __halves2half2(inp_data[1], inp_data[1]);
                vals_f[2] = __halves2half2(inp_data[2], inp_data[2]);
                vals_f[3] = __halves2half2(inp_data[3], inp_data[3]);
            }
            if (col_index < output_size) {
                __half2 weight_h[loop_unroll];
#pragma unroll
                for (int k = 0; k < loop_unroll; k++) weight_h[k] = weight_cast[k * output_size];
#pragma unroll
                for (int k = 0; k < (loop_unroll >> inner_loop_unroll_bits); k++)
#pragma unroll
                    for (int li = 0; li < inner_loop_unroll; li++) {
                        weight_h[0] = (vals_f[li] * weight_h[li]);
                        sum += weight_h[0];
                    }
            }
            wid += (warp_num << loop_unroll_bits);
            weight_cast += (output_size * (warp_num << loop_unroll_bits));
        }
    }
    {
        const __half2* bias_cast;
        if (bias) bias_cast = reinterpret_cast<const __half2*>(bias);
        __shared__ __half2 partial_result[MAX_WARP_NUM * (WARP_SIZE + 1)];

        partial_result[gid * (WARP_SIZE + 1) + lane] = sum;

        b.sync();
        float2 sum_f;
        sum_f = __half22float2(partial_result[lane * (WARP_SIZE + 1) + gid]);

        b.sync();
#pragma unroll
        for (int i = 1; i < WARP_SIZE; i *= 2) {
            sum_f.x += g.shfl_xor(sum_f.x, i);
            sum_f.y += g.shfl_xor(sum_f.y, i);
        }

        if (lane == 0) { partial_result[gid] = __float22half2_rn(sum_f); }

        b.sync();
        int bias_offset = output_size - intm_size;
        if (gid == 0) {
            int col = blockIdx.x * WARP_SIZE + lane;
            if (col < output_size) {
                sum = partial_result[lane];
                if (col >= bias_offset) {
                    float2 bias_f = __half22float2(bias_cast[col - bias_offset]);
                    sum_f = __half22float2(sum);
                    sum_f.x += bias_f.x;
                    sum_f.y += bias_f.y;
                    sum_f.x = gelu(sum_f.x);
                    sum_f.y = gelu(sum_f.y);
                    sum = __float22half2_rn(sum_f);
                    intm_cast[col - bias_offset] = sum;
                } else
                    output_cast[col] = sum;
            }
        }
    }
#endif
}

__global__ void input_tiled_gemm_kernel(float* output,
                                        float* intermediate,
                                        const float* vals,
                                        const float* weight,
                                        const float* gamma,
                                        const float* beta,
                                        float epsilon,
                                        const float* bias,
                                        int hidden_dim,
                                        int input_size,
                                        int output_size,
                                        int intm_size)
{
}

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
                                    cudaStream_t stream)
{
    constexpr int threads = 1024;
    output_size /= 2;
    dim3 grid_dim((output_size - 1) / WARP_SIZE + 1);
    dim3 block_dim(threads);
    input_tiled_gemm_kernel<<<grid_dim, block_dim, 0, stream>>>(output,
                                                                intermediate,
                                                                vals,
                                                                weight,
                                                                gamma,
                                                                beta,
                                                                epsilon,
                                                                bias,
                                                                hidden_dim,
                                                                input_size,
                                                                output_size,
                                                                intm_size / 2);
}

template void launch_input_tiled_gemm_kernel(float* output,
                                             float* intermediate,
                                             const float* vals,
                                             const float* weight,
                                             const float* gamma,
                                             const float* beta,
                                             float epsilon,
                                             const float* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             int intm_size,
                                             cudaStream_t stream);

template void launch_input_tiled_gemm_kernel(__half* output,
                                             __half* intermediate,
                                             const __half* vals,
                                             const __half* weight,
                                             const __half* gamma,
                                             const __half* beta,
                                             float epsilon,
                                             const __half* bias,
                                             int hidden_dim,
                                             int input_size,
                                             int output_size,
                                             int intm_size,
                                             cudaStream_t stream);
