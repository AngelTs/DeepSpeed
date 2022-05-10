#include "custom_cuda_layers.h"
#include <cuda_profiler_api.h>
namespace cg = cooperative_groups;

__global__ void transform_scale(float* query, float* kv_cache,
                                const float* vals,
                                int hidden_dim,
                                int seq_length,
                                unsigned cur_tokens,
                                uint64_t value_offset,
                                int heads,
                                int head_ext,
                                float norm_factor)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)
    kv_cache += d0 * (hidden_dim * cur_tokens);
    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(cnt == 0 ? query : (cnt == 1 ? kv_cache : kv_cache + value_offset));

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];

    inputs.x = (cnt < 2) ? inputs.x  * norm_factor : inputs.x;
    inputs.y = (cnt < 2) ? inputs.y  * norm_factor : inputs.y;
    inputs.z = (cnt < 2) ? inputs.z  * norm_factor : inputs.z;
    inputs.w = (cnt < 2) ? inputs.w  * norm_factor : inputs.w;
    
    output_vec[d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = inputs;
}

__global__ void transform_scale(__half* query, __half* kv_cache,
                                const __half* vals,
                                unsigned hidden_dim,
                                int seq_length,
                                unsigned cur_tokens,
                                uint64_t value_offset,
                                int heads,
                                int head_ext,
                                float norm_factor)
{


    //int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    //int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : MAX_OUT_TOKES);

    

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    //__half2* bias_half = reinterpret_cast<__half2*>(&bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    //const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(cnt == 0 ? query : (cnt == 1 ? kv_cache : kv_cache + value_offset));

    //vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    //bias_vec += (cnt * d1_stride);
    //bias_vec += (d2 * d2_stride);

    //output_vec += (d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    //output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    //bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

    output_half[0] = vals_half[0] ;//+ bias_half[0];
    output_half[1] = vals_half[1] ;//+ bias_half[1];
    output_half[2] = vals_half[2] ;//+ bias_half[2];
    output_half[3] = vals_half[3] ;//+ bias_half[3];
    output_vec[d3] = output_arr;
/*
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;


    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : MAX_OUT_TOKES);
    float4 vals_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);
    __half2 norm_factor_h = __float2half2_rn(norm_factor);
    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(cnt == 0 ? query : (cnt == 1 ? kv_cache : (kv_cache + value_offset)));

    if (cnt > 0) output_vec += d0 * (hidden_dim * cur_tokens);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    vals_arr = vals_vec[d3];

    output_half[0] = (cnt < 2) ? vals_half[0] * norm_factor_h : vals_half[0];
    output_half[1] = (cnt < 2) ? vals_half[1] * norm_factor_h : vals_half[1];
    output_half[2] = (cnt < 2) ? vals_half[2] * norm_factor_h : vals_half[2];
    output_half[3] = (cnt < 2) ? vals_half[3] * norm_factor_h : vals_half[3];
        
    output_vec[d3] = output_arr;
*/

}

// [B S C*H] - > C * [B A S N]
template <>
void launch_transform_scale<float>(float* vals,
                                   float* query,
                                   float* kv_cache,
                                   int batch_size,
                                   int seq_length,
                                   unsigned cur_tokens,
                                   size_t value_offset,
                                   unsigned hidden_dim,
                                   int heads,
                                   cudaStream_t stream,
                                   int trans_count,
                                   float norm_factor)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    transform_scale<<<grid_dim, block_dim, 0, stream>>>(
        query, kv_cache, vals, hidden_dim, seq_length, cur_tokens, value_offset, heads, head_ext, norm_factor);
}

template <>
void launch_transform_scale<__half>(__half* vals,
                                    __half* query,
                                    __half* kv_cache,
                                    int batch_size,
                                    int seq_length,
                                    unsigned cur_tokens,
                                    size_t value_offset,
                                    unsigned hidden_dim,
                                    int heads,
                                    cudaStream_t stream,
                                    int trans_count,
                                    float norm_factor)
{
    hidden_dim >>= 3;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
    transform_scale<<<grid_dim, block_dim, 0, stream>>>(
        query, kv_cache, vals, hidden_dim, seq_length, cur_tokens, value_offset,  heads, head_ext, norm_factor);
}

// Bias add

__global__ void bias_add_transform_0213(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];
    float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    float4 outputs;
    outputs.x = inputs.x + biases.x;
    outputs.y = inputs.y + biases.y;
    outputs.z = inputs.z + biases.z;
    outputs.w = inputs.w + biases.w;

    output_vec[cnt * d0_out_stride * gridDim.x + d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = outputs;
}

#define ATTN_H 3
#define MAX_SEQ_LINE 10

__global__ void bias_add_transform_0213(__half* output, //q
                                                __half* k_cache,
                                                __half* v_cache,
                                                const __half* vals, //qkv
                                                const __half* bias,
                                                int hidden_dim,
                                                int seq_length,
                                                unsigned seq_offset,
                                                int all_tokens,
                                                int heads,
                                                int rotary_dim,
                                                bool rotate_half,
                                                bool rotate_every_two,
                                                int head_ext)
{
#if __CUDA_ARCH__ >= 700

    unsigned half_dim = (rotary_dim << 3) >> 1;
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                   // Sequence ID (0-127)
    int cnt = blockIdx.z;                                                  // Hidden count
    int d2 = threadIdx.y;                                                  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)


    int d2_out_stride = d2_stride * (cnt == 0 ? seq_length : MAX_OUT_TOKES);
    float4 vals_arr;
    float4 output_arr;
    
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    float4* output_vec = reinterpret_cast<float4*>(cnt == 0 ? output : (cnt == 1 ? k_cache : v_cache));

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    unsigned seq_id = d1 + seq_offset;

    int lane = d3 & 0x1f;
    if (cnt < 2 && rotary_dim > 0 && d3 < rotary_dim) {
        float4 q = vals_vec[d3]; 
        __half2* q_h = reinterpret_cast<__half2*>(&q);
        if (rotate_every_two){
            #pragma unroll
            for (int o = 0;o < 4;o++){
                float inv_freq = (float)(((d3 << 2) + o) * 2) / (float)(rotary_dim << 3);
                inv_freq = 1.0 / powf(10000.0, inv_freq) * (float)seq_id;
                float q_data[2];
                q_data[0] = (float)q_h[o].x;
                q_data[1] = (float)q_h[o].y;
                q_h[o].x = (__half)(-1.0 * q_data[1] * sinf(inv_freq) + q_data[0] * cosf(inv_freq));
                q_h[o].y = (__half)(q_data[0] * sinf(inv_freq) + q_data[1] * cosf(inv_freq));
            }
        }
        //else{
        //    float4 q = vals_vec[d3]; 
        //    float4 q_rot;
        //    __half2* q_h = reinterpret_cast<__half2*>(&q);
        //    {
        //        __half2* qrot_h = reinterpret_cast<__half2*>(&q_rot);
        //        #pragma unroll
        //        for (int o = 0;o < 4;o++){
        //            int index = ((d3 << 2) + o) * 2;
        //            float rotary_sign = (index > (half_dim - 1) ? -1.0 : 1.0);
        //            qrot_h[o].x = __float2half((float)q_h[o].x * rotary_sign);
        //            qrot_h[o].y = __float2half((float)q_h[o].y * rotary_sign);
        //        }
        //    }
        //    __half2* qrot_h = reinterpret_cast<__half2*>(&q_rot);
        //    {
        //        float4 q_rot_tmp;
        //        float4 q_rot_tmp1;
        //        {
        //            q_rot_tmp.x = __shfl_sync(0xffffffff, q_rot.x, 1);
        //            q_rot_tmp.y = __shfl_sync(0xffffffff, q_rot.y, 1);
        //            q_rot_tmp.z = __shfl_sync(0xffffffff, q_rot.z, 1);
        //            q_rot_tmp.w = __shfl_sync(0xffffffff, q_rot.w, 1);
        //        }
        //        
        //        if(lane < 2){
        //            q_rot_tmp.x = __shfl_xor_sync(0xffffffff, q_rot.x, 1);
        //            q_rot_tmp.y = __shfl_xor_sync(0xffffffff, q_rot.y, 1);
        //            q_rot_tmp.z = __shfl_xor_sync(0xffffffff, q_rot.z, 1);
        //            q_rot_tmp.w = __shfl_xor_sync(0xffffffff, q_rot.w, 1);
        //        }
//
        //        {
        //            q_rot_tmp1.x = __shfl_sync(0xffffffff, q_rot.x, 2);
        //            q_rot_tmp1.y = __shfl_sync(0xffffffff, q_rot.y, 2);
        //            q_rot_tmp1.z = __shfl_sync(0xffffffff, q_rot.z, 2);
        //            q_rot_tmp1.w = __shfl_sync(0xffffffff, q_rot.w, 2);
        //        }
        //        if (lane %2 == 0){
        //            q_rot_tmp1.x = __shfl_xor_sync(0xffffffff, q_rot.x, 2);
        //            q_rot_tmp1.y = __shfl_xor_sync(0xffffffff, q_rot.y, 2);
        //            q_rot_tmp1.z = __shfl_xor_sync(0xffffffff, q_rot.z, 2);
        //            q_rot_tmp1.w = __shfl_xor_sync(0xffffffff, q_rot.w, 2);
        //        }
        //        q_rot.x = (lane < 1) ? q_rot_tmp.z : q_rot_tmp1.z;
        //        q_rot.y = (lane < 1) ? q_rot_tmp.w : q_rot_tmp1.w;
        //        q_rot.z = (lane > 0) ? q_rot_tmp.x : q_rot_tmp1.x;
        //        q_rot.w = (lane > 0) ? q_rot_tmp.y : q_rot_tmp1.y;
        //    }
        //    #pragma unroll
        //    for (int o = 0;o < 4;o++){
        //        int index = ((d3 << 2) + o) * 2;
        //        float inv_freq[2];
        //        inv_freq[0] = (float)((index % half_dim) * 2) / (float)(rotary_dim << 3);
        //        inv_freq[1] = (float)(((index+1) % half_dim) * 2) / (float)(rotary_dim << 3);
        //        inv_freq[0] = 1.0 / powf(10000.0, inv_freq[0]) * (float)seq_id;
        //        inv_freq[1] = 1.0 / powf(10000.0, inv_freq[1]) * (float)seq_id;
        //        q_h[o].x = (__half)((float)qrot_h[o].x * sinf(inv_freq[0]) + (float)q_h[o].x * cosf(inv_freq[0]));
        //        q_h[o].y = (__half)((float)qrot_h[o].y * sinf(inv_freq[1]) + (float)q_h[o].y * cosf(inv_freq[1]));
        //    }
        //}
        output_vec[d3] = q;
    }
    else 
        output_vec[d3] = vals_vec[d3];

#endif
}


// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                            float* k_cache, float* v_cache,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           unsigned seq_offset,
                                           int all_tokens,
                                           int hidden_dim,
                                           int heads,
                                           int rotary_dim,
                                           bool rotate_half,
                                           bool rotate_every_two,
                                           cudaStream_t stream,
                                           int trans_count)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<<<grid_dim, block_dim, 0, stream>>>(
        output, vals, bias, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_bias_add_transform_0213<__half>(__half* output,
                                            __half* k_cache, 
                                            __half* v_cache,
                                            const __half* vals,
                                            const __half* bias,
                                            int batch_size,
                                            int seq_length,
                                            unsigned seq_offset,
                                            int all_tokens,
                                            int hidden_dim,
                                            int heads,
                                            int rotary_dim,
                                            bool rotate_half,
                                            bool rotate_every_two,
                                            cudaStream_t stream,
                                            int trans_count)
{
    hidden_dim >>= 3;
    int head_ext = 1;// (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(1, seq_length, (trans_count * head_ext));
    bias_add_transform_0213<<<grid_dim, block_dim, 0, stream>>>(
        output, k_cache, v_cache, vals, bias, hidden_dim, seq_length, seq_offset,
        all_tokens, heads, 
        rotary_dim >> 3,
        rotate_half,
        rotate_every_two,
        head_ext);
}



// Bias add
template <typename T>
__global__ void bias_add_transform_0213(T* output,
                                        const T* vals,
                                        const T* bias,
                                        int hidden_dim,
                                        int seq_length,
                                        int heads,
                                        int head_ext);

template <>
__global__ void bias_add_transform_0213<float>(float* output,
                                               const float* vals,
                                               const float* bias,
                                               int hidden_dim,
                                               int seq_length,
                                               int heads,
                                               int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    float4 inputs = vals_vec[d0 * d0_stride * (gridDim.z / head_ext) + cnt * d1_stride +
                             d1 * d1_stride * (gridDim.z / head_ext) + d2 * d2_stride + d3];
    float4 biases = bias_vec[cnt * d1_stride + d2 * d2_stride + d3];

    float4 outputs;
    outputs.x = inputs.x + biases.x;
    outputs.y = inputs.y + biases.y;
    outputs.z = inputs.z + biases.z;
    outputs.w = inputs.w + biases.w;

    output_vec[cnt * d0_out_stride * gridDim.x + d0 * d0_out_stride + d1 * d1_out_stride +
               d2 * d2_out_stride + d3] = outputs;
}

template <>
__global__ void bias_add_transform_0213<__half>(__half* output,
                                                const __half* vals,
                                                const __half* bias,
                                                int hidden_dim,
                                                int seq_length,
                                                int heads,
                                                int head_ext)
{
#ifdef HALF_PRECISION_AVAILABLE

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = blockIdx.y;                                                  // Sequence ID (0-127)
    int cnt = blockIdx.z / head_ext;                                      // Hidden count
    int d2 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head (0-11)
    int d3 = threadIdx.x;                                                 // Values (groups of 4)

    float4 vals_arr;
    float4 bias_arr;
    float4 output_arr;
    __half2* vals_half = reinterpret_cast<__half2*>(&vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(&bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(&output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    vals_vec += (d0 * d0_stride * (gridDim.z / head_ext));
    vals_vec += (d1 * d1_stride * (gridDim.z / head_ext));
    vals_vec += (cnt * d1_stride);
    vals_vec += (d2 * d2_stride);

    bias_vec += (cnt * d1_stride);
    bias_vec += (d2 * d2_stride);

    output_vec += (cnt * d0_stride * gridDim.x);
    output_vec += (d1 * d2_stride);
    output_vec += (d0 * d0_stride);
    output_vec += (d2 * d2_out_stride);

    bias_arr = bias_vec[d3];
    vals_arr = vals_vec[d3];

    output_half[0] = vals_half[0] + bias_half[0];
    output_half[1] = vals_half[1] + bias_half[1];
    output_half[2] = vals_half[2] + bias_half[2];
    output_half[3] = vals_half[3] + bias_half[3];
    output_vec[d3] = output_arr;

#endif
}

__global__ void bias_add_transform_0213_v2(__half* output,
                                           const __half* vals,
                                           const __half* bias,
                                           int hidden_dim,
                                           int seq_length,
                                           int heads)
{
#ifdef HALF_PRECISION_AVAILABLE
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;
    int iteration_stride = d1_stride * blockDim.z;  // Hidden * 3 / 8
    int batch_stride = d0_stride * blockDim.z;      // Hidden * S * 3 / 8

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = d2_stride * seq_length;

    int d0 = blockIdx.x;    // Batch
    int d1 = blockIdx.y;    // Sequence ID (0-127)
    int cnt = threadIdx.z;  // blockIdx.z; // Hidden count
    int d2 = threadIdx.y;   // Head (0-11)
    int d3 = threadIdx.x;   // Values (groups of 4)

    float4 vals_arr[1];
    float4 bias_arr[1];
    float4 output_arr[1];
    __half2* vals_half = reinterpret_cast<__half2*>(vals_arr);
    __half2* bias_half = reinterpret_cast<__half2*>(bias_arr);
    __half2* output_half = reinterpret_cast<__half2*>(output_arr);

    const float4* vals_vec = reinterpret_cast<const float4*>(vals);
    const float4* bias_vec = reinterpret_cast<const float4*>(bias);
    float4* output_vec = reinterpret_cast<float4*>(output);

    int iter_index = cnt * d1_stride + d2 * d2_stride + d3;
    int input_offset = d0 * batch_stride + d1 * (iteration_stride << 1);
    bias_arr[0] = bias_vec[iter_index];

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        vals_arr[0] = vals_vec[input_offset + iter_id];

        output_half[0] = vals_half[0] + bias_half[0];
        output_half[1] = vals_half[1] + bias_half[1];
        output_half[2] = vals_half[2] + bias_half[2];
        output_half[3] = vals_half[3] + bias_half[3];

        in_data[iter_id] = output_arr[0];
    }
    __syncthreads();

    iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_out_stride * gridDim.x);
    int head_count = (d2 >> 1) + cnt * (blockDim.y >> 1);

    int out_index = d0 * d0_out_stride + d1 * (d1_out_stride << 1) + d3 + (d2 % 2) * d2_stride;

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = (iter * iteration_stride) + head_count;
        int iter_offset =
            (iter_row % blockDim.y) * d2_out_stride + (iter_row / blockDim.y) * matrix_stride;
        output_vec[out_index + iter_offset] =
            in_data[iter_row * d2_stride + d3 + (d2 % 2) * (d1_stride * blockDim.z)];
    }
#endif
}

// [B S C*H] - > C * [B A S N]
template <>
void launch_bias_add_transform_0213<float>(float* output,
                                           const float* vals,
                                           const float* bias,
                                           int batch_size,
                                           int seq_length,
                                           int hidden_dim,
                                           int heads,
                                           cudaStream_t stream,
                                           int trans_count)
{
    hidden_dim >>= 2;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;

    dim3 block_dim(hidden_dim / heads, (heads / head_ext));
    dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));

    bias_add_transform_0213<float><<<grid_dim, block_dim, 0, stream>>>(
        output, vals, bias, hidden_dim, seq_length, heads, head_ext);
}

template <>
void launch_bias_add_transform_0213<__half>(__half* output,
                                            const __half* vals,
                                            const __half* bias,
                                            int batch_size,
                                            int seq_length,
                                            int hidden_dim,
                                            int heads,
                                            cudaStream_t stream,
                                            int trans_count)
{
    hidden_dim >>= 3;
    if (true) { // (hidden_dim > 128 || hidden_dim < 16) {
        int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
        dim3 block_dim(hidden_dim / heads, (heads / head_ext));
        dim3 grid_dim(batch_size, seq_length, (trans_count * head_ext));
        bias_add_transform_0213<__half><<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads, head_ext);
    } else {
        dim3 block_dim(hidden_dim / heads, heads, trans_count);
        dim3 grid_dim(batch_size, seq_length / 2);
        bias_add_transform_0213_v2<<<grid_dim, block_dim, 0, stream>>>(
            output, vals, bias, hidden_dim, seq_length, heads);
    }
}

template <typename T>
__global__ void transform4d_0213(T* out,
                                 const T* in,
                                 int heads,
                                 int seq_length,
                                 int hidden_dim,
                                 int head_ext);

template <>
__global__ void transform4d_0213<float>(float* out,
                                        const float* in,
                                        int heads,
                                        int seq_length,
                                        int hidden_dim,
                                        int head_ext)
{
    int d0_stride = hidden_dim * seq_length;
    int d1_stride = d0_stride / heads;
    int d2_stride = hidden_dim / heads;

    int d0_out_stride = d0_stride;
    int d1_out_stride = d2_stride;
    int d2_out_stride = hidden_dim;

    int d0 = blockIdx.x;                                        // Batch
    int d1 = blockIdx.y / ((seq_length - 1) / blockDim.y + 1);  // Head
    int d2 = (threadIdx.y + blockDim.y * blockIdx.y) % seq_length;
    int cnt = blockIdx.z;
    int d3 = threadIdx.x;  // Values (groups of 8)

    if (d2 < seq_length) {
        const float4* in_vec = reinterpret_cast<const float4*>(in);
        float4* out_vec = reinterpret_cast<float4*>(out);

        float4 vals_vec = in_vec[cnt * d0_stride * gridDim.x + d0 * d0_stride + d1 * d1_stride +
                                 d2 * d2_stride + d3];
        out_vec[d0 * d0_out_stride * gridDim.z + cnt * d2_out_stride + d1 * d1_out_stride +
                d2 * d2_out_stride * gridDim.z + d3] = vals_vec;
    }
}

template <>
__global__ void transform4d_0213<__half>(__half* out,
                                         const __half* in,
                                         int heads,
                                         int seq_length,
                                         int hidden_dim,
                                         int head_ext)
{
#if __CUDA_ARCH__ >= 700

    int d0_stride = hidden_dim * (seq_length / head_ext);
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;                                                  // Batch
    int d1 = threadIdx.y + (blockIdx.z % head_ext) * (heads / head_ext);  // Head
    int d2 = blockIdx.z / head_ext;                                       // Sequence
    int cnt = blockIdx.y;                                                 // Hidden count
    int d3 = threadIdx.x;                                                 // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    in_vec += (cnt * d0_stride * gridDim.x);
    in_vec += (d0 * d0_stride);
    in_vec += (d2 * d2_stride);
    in_vec += (d1 * d2_stride * seq_length);

    out_vec += (cnt * d1_stride);
    out_vec += (d1 * d2_stride);
    out_vec += (d0 * d0_stride * gridDim.y);
    out_vec += (d2 * d1_stride * gridDim.y);

    out_vec[d3] = in_vec[d3];

#endif
}

__global__ void transform4d_0213_v2(__half* out,
                                    const __half* in,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim)
{
#if __CUDA_ARCH__ >= 700
    __shared__ float4 in_data[3072];

    int d0_stride = hidden_dim * seq_length;
    int d1_stride = hidden_dim;
    int d2_stride = hidden_dim / heads;

    int d0 = blockIdx.x;    // Batch
    int d1 = threadIdx.y;   // Head
    int d2 = blockIdx.y;    // Sequence
    int cnt = threadIdx.z;  // Hidden count
    int d3 = threadIdx.x;   // Values (groups of 8)

    const float4* in_vec = reinterpret_cast<const float4*>(in);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int input_offset = d0 * d0_stride + d2 * (d2_stride << 1) + d3 + (d1 % 2) * d2_stride;
    int head_count = (d1 >> 1) + cnt * (blockDim.y >> 1);
    int iteration_stride = blockDim.z * (blockDim.y >> 1);
    int matrix_stride = (d0_stride * gridDim.x);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_row = iter * iteration_stride + head_count;
        int iter_offset = (iter_row % blockDim.y) * d2_stride;

        in_data[d3 + iter_offset + (iter_row / blockDim.y + (d1 % 2) * blockDim.z) * d1_stride] =
            in_vec[input_offset + iter_offset * seq_length +
                   (iter_row / blockDim.y) * matrix_stride];
    }
    __syncthreads();

    iteration_stride = d1_stride * blockDim.z;
    int iter_index = cnt * d1_stride + d1 * d2_stride + d3;
    int output_offset = d0 * d0_stride * blockDim.z + d2 * (iteration_stride << 1);

#pragma unroll
    for (int iter = 0; iter < 2; iter++) {
        int iter_id = iter * iteration_stride + iter_index;
        out_vec[output_offset + iter_id] = in_data[iter_id];
    }
#endif
}

// 3 * [B A S N] - > [B S C*H]
template <>
void launch_transform4d_0213<float>(float* out,
                                    const float* in,
                                    int batch_size,
                                    int heads,
                                    int seq_length,
                                    int hidden_dim,
                                    cudaStream_t stream,
                                    int trans_count)
{
    hidden_dim >>= 2;
    dim3 grid_dims(batch_size, heads * ((seq_length - 1) / 8 + 1), trans_count);
    dim3 block_dims(hidden_dim / heads, 8);
    transform4d_0213<float>
        <<<grid_dims, block_dims, 0, stream>>>(out, in, heads, seq_length, hidden_dim, 1);
}

template <>
void launch_transform4d_0213<__half>(__half* out,
                                     const __half* in,
                                     int batch_size,
                                     int heads,
                                     int seq_length,
                                     int hidden_dim,
                                     cudaStream_t stream,
                                     int trans_count)
{
    hidden_dim >>= 3;
    int head_ext = (hidden_dim - 1) / MAX_THREADS + 1;
    dim3 grid_dims(batch_size, trans_count, (seq_length * head_ext));
    dim3 block_dims(hidden_dim / heads, (heads / head_ext));
    transform4d_0213<__half><<<grid_dims, block_dims, 0, stream>>>(
        out, in, heads, seq_length, hidden_dim, head_ext);
}
