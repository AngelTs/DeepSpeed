#include <cstdio>
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace dequant {
constexpr int store_granularity = 16;
constexpr int unroll = 4;
const int elems_per_store = store_granularity / sizeof(__half);
const int threads = 256;
}  // namespace dequant

template <int q_bits>
__global__ void dequantization(__half* output,
                               const int8_t* quantized_data,
                               const float* scales,
                               int elems_per_group,
                               int total_elems) {
    constexpr int load_granularity = (q_bits == 8) ? 8 : 4;
    constexpr int elems_per_load = load_granularity / sizeof(int8_t);
    const int load_stride = dequant::threads * elems_per_load;
    const int load_block_offset = load_stride * dequant::unroll * blockIdx.x;
    const int load_thread_offset = elems_per_load * threadIdx.x;
    const int load_offset = load_block_offset + load_thread_offset;
    const int8_t* base_quantized_data = quantized_data + load_offset;
    const int store_stride = dequant::threads * dequant::elems_per_store;
    const int store_block_offset = store_stride * dequant::unroll * blockIdx.x;
    const int store_thread_offset = dequant::elems_per_store * threadIdx.x;
    const int store_offset = store_block_offset + store_thread_offset;
    __half* base_output_data = output + store_offset;

#pragma unroll
    for (int i = 0; i < dequant::unroll; i++) {
        int8_t load_buffer[elems_per_load];
        const int iter_idx = store_offset + i * store_stride;
        mem_access::load_global<load_granularity>(
            load_buffer, base_quantized_data + load_stride * i, iter_idx < total_elems);
        float scale = 0.f;
        mem_access::load_global<sizeof(float)>(
            &scale, scales + iter_idx / elems_per_group, iter_idx < total_elems);
        __half store_buffer[dequant::elems_per_store];

        // Dequantize
        for (int j = 0; j < elems_per_load; j++) {
            if constexpr (q_bits == 8) {
                store_buffer[j] = __float2half(scale * load_buffer[j]);
            } else {
                auto data = *(int4x2_t*)(&load_buffer[j]);
                float val1 = scale * data.low;
                float val2 = scale * data.high;
                store_buffer[2*j] = __float2half(val1);
                store_buffer[2*j + 1] = __float2half(val2);

            }
        }
        if (iter_idx < total_elems) {
            mem_access::store_global<dequant::store_granularity>(
                base_output_data + store_stride * i, store_buffer);
        }
    }
}

void launch_dequant(__half* output,
                    const int8_t* quantized_data,
                    const float* scales,
                    int elems_per_group,
                    int total_elems,
                    cudaStream_t stream)
{
    const int blocks = (total_elems + dequant::threads * dequant::unroll * dequant::elems_per_store - 1) /
                       (dequant::threads * dequant::unroll * dequant::elems_per_store);
    dequantization<8><<<blocks, dequant::threads, 0, stream>>>(
        output, quantized_data, scales, elems_per_group, total_elems);
}

void launch_dequant_int4(__half* output,
                         const int8_t* quantized_data,
                         const float* scales,
                         int elems_per_group,
                         int total_elems,
                         cudaStream_t stream)
{
    const int blocks = (total_elems + dequant::threads * dequant::unroll * dequant::elems_per_store - 1) /
                       (dequant::threads * dequant::unroll * dequant::elems_per_store);
    dequantization<4><<<blocks, dequant::threads, 0, stream>>>(
        output, quantized_data, scales, elems_per_group, total_elems);
}