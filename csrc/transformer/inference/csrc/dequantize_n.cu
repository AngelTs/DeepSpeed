#include <cstdio>
#include "inference_cuda_layers.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"

namespace dequant {
constexpr int store_granularity = 16;
constexpr int unroll = 4;
const int elems_per_store = store_granularity / sizeof(__half);
const int threads = 256;
}  // namespace dequant

using rop = reduce::ROpType;

struct PackedInt4 {
    int8_t high : 4;
    int8_t low : 4;
};

/*
Modified from quantization utils, should be replaced
*/
template <int numBits>
DS_D_INLINE void quantize_chunk(int8_t* local_output, const __half2* data, float scale)
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

template <int numBits, int numTensors, int totalChunks>
__global__ void __launch_bounds__(1024) dequant_reduce(int8_t* reduced_data,
                               float* reduced_scales,
                               const int8_t* input_data,
                               const float* input_scales,
                               int elems_per_out_group,
                               int elems_per_in_tensor,
                               int groups_per_in_tensor,
                               int elems_per_in_group)
{
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp =
        cg::tiled_partition<hw_warp_size>(tb);

    // NOTE(cmikeh2): This probably could be hardcoded to a larger number,
    // but that means even stronger restrictions on the number of elements per group
    // A performance analysis here might be beneficial
    constexpr int mem_granularity = (numBits == 8) ? 8 : 4;
    constexpr int elems_per_load = mem_granularity / sizeof(int8_t); // div by 1
    constexpr int storage_values = 16 / sizeof(__half2);

    const int block_offset = tb.group_index().x * elems_per_out_group;
    const int elem_offset = tb.thread_index().x * elems_per_load;
    const int base_offset = block_offset + elem_offset;
    const int stride = tb.group_dim().x * elems_per_load;

    __half2 local_buffer[totalChunks * storage_values];

    __half2 max_h2 = reduce::init<rop::Max, __half2>();

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        __half2 * iteration_buffer = local_buffer + i * storage_values;

#pragma unroll
        for (int j = 0; j < storage_values; j++) {
            iteration_buffer[j] = reduce::init<rop::Add, __half2>();
        }

        const int iter_offset = i * stride + base_offset;
        const int iter_scale_idx = iter_offset / elems_per_in_group;
        bool do_loads = i * stride + elem_offset < elems_per_out_group;

#pragma unroll
        for (int j = 0; j < numTensors; j++) {
            if (do_loads) {
                int8_t load_buffer[elems_per_load];
                float scale;

                mem_access::load_global<mem_granularity>(
                    load_buffer, input_data + j * elems_per_in_tensor + iter_offset);
                mem_access::load_global<sizeof(float)>(
                    &scale, input_scales + j * groups_per_in_tensor + iter_scale_idx);

                // Dequantize
                for (int k = 0; k < storage_values; k++) {
                    if constexpr (numBits == 8) {
                        float2 raw_val;
                        raw_val.x = conversion::to<float>(load_buffer[2 * k]) * scale;
                        raw_val.y = conversion::to<float>(load_buffer[2 * k + 1]) * scale;
                        __half2 dequant_data = conversion::to<__half2>(raw_val);
                        iteration_buffer[k] = reduce::element<rop::Add>(iteration_buffer[k], dequant_data);
                    } else {
                        auto data = *(int4x2_t*)(&load_buffer[k]);
                        float2 raw_val;
                        raw_val.x = scale * conversion::to<float>(data.low);
                        raw_val.y = scale * conversion::to<float>(data.high);
                        __half2 dequant_data = conversion::to<__half2>(raw_val);
                        iteration_buffer[k] = reduce::element<rop::Add>(iteration_buffer[k], dequant_data);
                    }
                }
            }
        }

        for (int j = 0; j < storage_values; j++) {
            __half2 abs_vals = __habs2(iteration_buffer[j]);
            max_h2 = reduce::element<rop::Max>(max_h2, abs_vals);
        }
    }

    const __half max_h = reduce::element<rop::Max>(max_h2.x, max_h2.y);
    float max = conversion::to<float>(max_h);
    reduce::block<rop::Max>(tb, warp, max);

    // We add an extra scaling factor here to perform the averaging reduction.
    const float scale = (1 << numBits) / (2 * (max / numTensors));
    if (tb.thread_index().x == 0) reduced_scales[tb.group_index().x] = 1 / scale;

#pragma unroll
    for (int i = 0; i < totalChunks; i++) {
        const int iter_offset = i * stride + base_offset;
        if (i * stride + elem_offset < elems_per_out_group) {
            int8_t local_output[elems_per_load];
            quantize_chunk<numBits>(
                local_output, local_buffer + i * storage_values, scale / numTensors);
            mem_access::store_global<mem_granularity>(
                reduced_data + iter_offset, local_output);
        }
    }
}

template <int Power>
int32_t pow2_round(int32_t raw_value) { return (((raw_value - 1) >> Power) + 1) << Power; }

#define LAUNCH_DEQUANT_REDUCE(num_chunks) \
    dequant_reduce<numBits, numTensors, num_chunks>             \
        <<<grid, block, 0, stream>>>(reduced_data,              \
                                     reduced_scales,            \
                                     input_data,                \
                                     input_scales,              \
                                     elems_per_out_group,       \
                                     elems_per_in_tensor,       \
                                     groups_per_in_tensor,      \
                                     elems_per_in_group);

template <int numBits, int numTensors>
void launch_dequant_reduce_impl(int8_t* reduced_data,
                                float* reduced_scales,
                                const int8_t* input_data,
                                const float* input_scales,
                                int out_groups,
                                int elems_per_out_group,
                                int elems_per_in_tensor,
                                int groups_per_in_tensor,
                                int elems_per_in_group,
                                cudaStream_t stream)
{
    // This is a coincidence. This is derived by 8 halves per 16 bytes with 2-way packing for int4
    constexpr int elems_per_thread = numBits;
    const int one_step_threads = pow2_round<5>((elems_per_out_group + elems_per_thread - 1) / (elems_per_thread));
    // TODO(cmikeh2): Tune this
    const int threads = (one_step_threads < 1024) ? one_step_threads: 1024;

    dim3 block(threads);
    dim3 grid(out_groups);

    const int elems_per_step = threads * elems_per_thread;
    const int unroll_raw = (elems_per_out_group + elems_per_step - 1) / elems_per_step;

    const int unroll = (unroll_raw >= 4) ? pow2_round<1>(unroll_raw) : unroll_raw;

    if (unroll == 1) {
        // 0-4096 elems
        LAUNCH_DEQUANT_REDUCE(1);
    } else if (unroll == 2) {
        // 4097-8192
        LAUNCH_DEQUANT_REDUCE(2);
    } else if (unroll == 3) {
        LAUNCH_DEQUANT_REDUCE(3);
    } else if (unroll == 4) {
        LAUNCH_DEQUANT_REDUCE(4);
    } else if (unroll == 6) {
        LAUNCH_DEQUANT_REDUCE(6);
    } else if (unroll == 8) {
        LAUNCH_DEQUANT_REDUCE(8);
    } else if (unroll == 10) {
        LAUNCH_DEQUANT_REDUCE(10);
    } /*else if (unroll == 12) {
        LAUNCH_DEQUANT_REDUCE(12);
    } else if (unroll == 14) {
        LAUNCH_DEQUANT_REDUCE(14);
    } else if (unroll == 16) {
        LAUNCH_DEQUANT_REDUCE(16);
    } else if (unroll == 18) {
        LAUNCH_DEQUANT_REDUCE(18);
    } else if (unroll == 20) {
        // 80k maximum
        LAUNCH_DEQUANT_REDUCE(20);
    }*/
}

#define LAUNCH_DEQUANT_REDUCE_IMPL(NUM_BITS, NUM_GPUS)                  \
    launch_dequant_reduce_impl<NUM_BITS, NUM_GPUS>(reduced_data,             \
                                                   reduced_scales,           \
                                                   input_data,               \
                                                   input_scales,             \
                                                   out_groups,               \
                                                   elems_per_out_group,      \
                                                   elems_per_in_tensor,      \
                                                   groups_per_in_tensor,     \
                                                   elems_per_in_group,       \
                                                   stream);

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
                           cudaStream_t stream)
{
    if (num_bits == 4 && num_gpus == 4) {
        //LAUNCH_DEQUANT_REDUCE_IMPL(4, 4);
    } else if (num_bits == 4 && num_gpus == 8) {
        LAUNCH_DEQUANT_REDUCE_IMPL(4, 8);
    } else if (num_bits == 4 && num_gpus == 16) {
        LAUNCH_DEQUANT_REDUCE_IMPL(4, 16);
    } else if (num_bits == 8 && num_gpus == 4) {
       // LAUNCH_DEQUANT_REDUCE_IMPL(8, 4);
    } else if (num_bits == 8 && num_gpus == 8) {
        //LAUNCH_DEQUANT_REDUCE_IMPL(8, 8);
    }
}

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