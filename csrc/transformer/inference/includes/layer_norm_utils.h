/*
Copyright 2022 The Microsoft DeepSpeed Team
*/

#include "ds_kernel_utils.h"

namespace ln {
constexpr int granularity = 16;
constexpr int max_threads = 256;
constexpr int max_warps = max_threads / hw_warp_size;

constexpr int internal_unroll = 4;
}  // namespace ln

namespace cg = cooperative_groups;

template <int MAX_WARPS>
__device__ __forceinline__ float ln_sum_reduce(cg::thread_block& tb,
                                               cg::thread_block_tile<hw_warp_size>& warp,
                                               float* sum_buffer,
                                               float partial_sum_arg)
{
    float partial_sum = partial_sum_arg;

#pragma unroll
    for (int i = hw_warp_size / 2; i > 0; i /= 2) { partial_sum += warp.shfl_down(partial_sum, i); }

    if (warp.meta_group_size() > 1) {
        // If more than one warp, broadcast partial sums, have one warp reduce and broadcast
        if (warp.thread_rank() == 0) sum_buffer[warp.meta_group_rank()] = partial_sum;

        tb.sync();

        if (warp.meta_group_rank() == 0) {
            float r_sum = 0.f;
            if (warp.thread_rank() < warp.meta_group_size()) {
                r_sum = sum_buffer[warp.thread_rank()];
            }

            // TODO(cmikeh2): this is an over-parametrized reduction, but the max_warps helps
            // save some overhead if we can bound the block size we want to launch with
#pragma unroll
            for (int i = MAX_WARPS / 2; i > 0; i /= 2) { r_sum += warp.shfl_down(r_sum, i); }

            if (warp.thread_rank() == 0) sum_buffer[0] = r_sum;
        }

        tb.sync();
        return sum_buffer[0];
    } else {
        // If one warp, we can just use a warp broadcast and continue
        return warp.shfl(partial_sum, 0);
    }
}
