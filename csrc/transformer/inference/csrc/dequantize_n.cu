#include <cstdio>
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"

namespace cg = cooperative_groups;

namespace dequant {
constexpr int store_granularity = 16;
constexpr int load_granularity = 8;
constexpr int unroll = 4;
const int elems_per_store = store_granularity / sizeof(__half);
const int elems_per_load = load_granularity / sizeof(int8_t);
const int threads = 256;
}  // namespace dequant

__global__ void dequantization(__half* output,
                               const int8_t* quantized_data,
                               const float* scales,
                               int elems_per_group,
                               int total_elems)
{
    const int load_stride = dequant::threads * dequant::elems_per_load;
    const int load_block_offset = load_stride * dequant::unroll * blockIdx.x;
    const int load_thread_offset = dequant::elems_per_load * threadIdx.x;
    const int load_offset = load_block_offset + load_thread_offset;
    const int8_t* base_quantized_data = quantized_data + load_offset;
    const int store_stride = dequant::threads * dequant::elems_per_store;
    const int store_block_offset = store_stride * dequant::unroll * blockIdx.x;
    const int store_thread_offset = dequant::elems_per_store * threadIdx.x;
    const int store_offset = store_block_offset + store_thread_offset;
    __half* base_output_data = output + store_offset;

#pragma unroll
    for (int i = 0; i < dequant::unroll; i++) {
        int8_t load_buffer[dequant::elems_per_load];
        const int iter_idx = store_offset + i * store_stride;
        mem_access::load_global<dequant::load_granularity>(
            load_buffer, base_quantized_data + load_stride * i, iter_idx < total_elems);
        float scale = 0.f;
        mem_access::load_global<sizeof(float)>(
            &scale, scales + iter_idx / elems_per_group, iter_idx < total_elems);
        __half store_buffer[dequant::elems_per_store];

        // Dequantize
        for (int j = 0; j < dequant::elems_per_load; j++) {
            store_buffer[j] = __float2half(scale * load_buffer[j]);
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
    const int blocks = (total_elems + dequant::threads * dequant::unroll - 1) /
                       (dequant::threads * dequant::unroll);
    dequantization<<<blocks, dequant::threads, 0, stream>>>(
        output, quantized_data, scales, elems_per_group, total_elems);
}