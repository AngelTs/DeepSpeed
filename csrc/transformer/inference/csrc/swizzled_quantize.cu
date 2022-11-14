
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

using rop = reduce::ROpType;

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

namespace swiz_quant {
    constexpr int load_granularity = 16;
    constexpr int h2_per_load = load_granularity / sizeof(__half2);
    constexpr int h_per_load = load_granularity / sizeof(__half);

    constexpr int threads = 512;
    constexpr int step_granularity = 2;
    constexpr int h_per_step = step_granularity * h_per_load;
}

template <int numBits>
DS_D_INLINE void quantize_chunk_s(int8_t* local_output, const __half2* data, float scale)
{
    constexpr int32_t elems = 16 / sizeof(__half);
    constexpr int32_t num_elems_packed = 8 / numBits;
    constexpr int32_t q_min = -(1 << (numBits - 1));
    constexpr int32_t q_max = (1 << (numBits - 1)) - 1;

    const __half* data_h = reinterpret_cast<const __half*>(data);

#pragma unroll
    for (int i = 0, oi = 0; i < elems; i += num_elems_packed, oi++) {
        if (num_elems_packed == 1) {
            float data_f = conversion::to<float>(data_h[i]) * scale;
            int32_t data_i32 = conversion::to<int>(data_f);
            data_i32 = min(max(data_i32, q_min), q_max);
            local_output[i] = (int8_t)data_i32;
        } else if (num_elems_packed == 2) {
            float data_f_1 = conversion::to<float>(data_h[i]) * scale;
            float data_f_2 = conversion::to<float>(data_h[i + 1]) * scale;
            int32_t data_i32_1 = conversion::to<int32_t>(data_f_1);
            int32_t data_i32_2 = conversion::to<int32_t>(data_f_2);
            int8_t data_i8_1 = (int8_t)min(max(data_i32_1, q_min), q_max);
            int8_t data_i8_2 = (int8_t)min(max(data_i32_2, q_min), q_max);
            auto data_i8 = PackedInt4{data_i8_2, data_i8_1};
            local_output[oi] = *((int8_t*)(&data_i8));
        }
    }
}

template <int numBits, int totalChunks>
__global__ void swizzled_quant_kernel(int8_t* quantized_data,
                                      float* quantized_scales,
                                      const __half* uncompressed_data,
                                      int elems_per_group,
                                      int nodes,
                                      int devices_per_node)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp =
        cg::tiled_partition<hw_warp_size>(tb);

    // Indexing offsets, same as normal quantization for in-case
    const int block_rank = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    const int block_offset = block_rank * elems_per_group;
    const int elem_offset = tb.thread_index().x * swiz_quant::h_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.size() * swiz_quant::h_per_load;
    const __half* input_base = uncompressed_data + base_offset;

    // Local buffer
    __half2 local_buffer[totalChunks * swiz_quant::h2_per_load];

    __half2 max_h2 = reduce::init<rop::Max, __half2>();

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        __half2 * iteration_buffer = local_buffer + i * swiz_quant::h2_per_load;

        mem_access::load_global<swiz_quant::load_granularity>(
            iteration_buffer,
            input_base + i * stride,
            elem_offset + i * stride < elems_per_group);

#pragma unroll
        for (int j = 0; j < swiz_quant::h2_per_load; j++) {
            __half2 abs_vals = __habs2(iteration_buffer[j]);
            max_h2 = reduce::element<rop::Max>(max_h2, abs_vals);
        }
    }

    const __half max_h = reduce::element<rop::Max>(max_h2.x, max_h2.y);
    float max = conversion::to<float>(max_h);
    reduce::block<rop::Max>(tb, warp, max);

    const float scale = (1 << numBits) / (2 * max);

    const int partition_id = blockIdx.z;
    const int partition_offset = partition_id / devices_per_node;
    const int partition_base = (partition_id % devices_per_node) * nodes;
    const int pipelining_offset = blockIdx.y * (devices_per_node * nodes);
    const int output_partition = (pipelining_offset + partition_base + partition_offset);

    constexpr int out_scalar_effect = 8 / numBits;
    const int out_block_rank = output_partition * gridDim.x + blockIdx.x;
    const int out_block_offset = out_block_rank * elems_per_group / out_scalar_effect;
    const int out_base_offset = out_block_offset + elem_offset / out_scalar_effect;
    int8_t * out_base = quantized_data + out_base_offset;

    const int out_stride = stride / out_scalar_effect;

    if (tb.thread_index().x == 0) quantized_scales[out_block_rank] = 1 / scale;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        if (i * stride + elem_offset < elems_per_group) {
            int8_t local_output[swiz_quant::h_per_load / out_scalar_effect];
            quantize_chunk_s<numBits>(
                local_output, local_buffer + i * swiz_quant::h2_per_load, scale);
            mem_access::store_global<swiz_quant::load_granularity / (2 * out_scalar_effect)>(
                out_base + i * out_stride, local_output);
        }
    }
}

int32_t round_to_32_s(int32_t raw_value) { return (((raw_value - 1) >> 5) + 1) << 5; }

#define LAUNCH_SWIZZLE_QUANT(num_bits, total_chunks)                    \
    swizzled_quant_kernel<numBits, total_chunks>       \
        <<<grid, block, 0, stream>>>(q_data,            \
                                     q_scales,          \
                                     input_data,        \
                                     elems_per_group,   \
                                     nodes,             \
                                     devices_per_node);

/*
Swizzled quantization reorganizes the quantized groups in order to better facilitate
communication. As an example of the partioning scheme we have the following example
of 2 node, 4 device swizzling:

 --- --- --- --- --- --- --- ---
| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
 --- --- --- --- --- --- --- ---
becomes
 --- --- --- --- --- --- --- ---
| 0 | 4 | 1 | 5 | 2 | 6 | 3 | 7 |
 --- --- --- --- --- --- --- ---

Multiple quantization groups may be mapped into a single partition. In order to better support
later pipelining, we may also perform an additional slicing. In two-way slicing, for instance,
the first halves of each partition are concatenated.
*/

template <int numBits>
void launch_swizzled_quant_impl(int8_t* q_data,
                                float* q_scales,
                                const __half* input_data,
                                int groups,
                                int elems_per_group,
                                int pipelining,
                                int nodes,
                                int devices_per_node,
                                cudaStream_t stream)
{
    const int one_step_threads =
        round_to_32_s((elems_per_group + swiz_quant::h_per_step - 1) / swiz_quant::h_per_step);
    const int threads = (one_step_threads < swiz_quant::threads) ? one_step_threads
                                                                : swiz_quant::threads;

    dim3 block(threads);
    const int groups_per_partition = groups / (nodes * devices_per_node);
    assert (groups_per_partition % pipelining == 0);
    const int contiguous_groups = groups_per_partition / pipelining;
    const int partitions = nodes * devices_per_node;
    dim3 grid(contiguous_groups, pipelining, partitions);

    const int elems_per_step = threads * swiz_quant::h_per_step;
    const int external_unroll = ((elems_per_group + elems_per_step - 1) / elems_per_step);
    const int total_unroll = external_unroll * swiz_quant::step_granularity;

    assert (total_unroll % 2 == 0);

    if (total_unroll == 2) {
        LAUNCH_SWIZZLE_QUANT(numBits, 2);
    } else if (total_unroll == 4) {
        LAUNCH_SWIZZLE_QUANT(numBits, 4);
    } else if (total_unroll == 6) {
        LAUNCH_SWIZZLE_QUANT(numBits, 6);
    } else if (total_unroll == 8) {
        LAUNCH_SWIZZLE_QUANT(numBits, 8);
    } else if (total_unroll == 10) {
        LAUNCH_SWIZZLE_QUANT(numBits, 10);
    }
}

void launch_swizzled_quant(int8_t* q_data,
                           float* q_scales,
                           const __half* input_data,
                           int num_bits,
                           int groups,
                           int elems_per_group,
                           int pipelining,
                           int nodes,
                           int devices_per_node,
                           cudaStream_t stream)
{
    if (num_bits == 4) {
        launch_swizzled_quant_impl<4>(q_data,
                                      q_scales,
                                      input_data,
                                      groups,
                                      elems_per_group,
                                      pipelining,
                                      nodes,
                                      devices_per_node,
                                      stream);
    } else if (num_bits == 8) {
        launch_swizzled_quant_impl<8>(q_data,
                                      q_scales,
                                      input_data,
                                      groups,
                                      elems_per_group,
                                      pipelining,
                                      nodes,
                                      devices_per_node,
                                      stream);
    }
}
