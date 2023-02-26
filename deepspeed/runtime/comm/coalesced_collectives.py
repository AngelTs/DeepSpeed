"""batched collective operations for overhead amortization and better
bandwidth utilization"""

import math
from typing import List
import time
import torch
from torch import Tensor
from deepspeed import comm as dist
# NOTE: Use torch.distributed's ProcessGroup class until we have our own.
from torch.distributed import ProcessGroup, all_to_all_single

from deepspeed.utils import instrument_w_nvtx
#from torch._C._distributed_c10d import AllToAllOptions, ReduceScatterOptions,ReduceOp
from deepspeed.utils import groups
from deepspeed.ops import op_builder
quantizer_cuda_module = op_builder.InferenceBuilder().load()

def _torch_reduce_scatter_fn(input_tensor: Tensor,
                             output_tensor: Tensor,
                             group=None,
                             async_op=False,
                             prof=False):
    return instrument_w_nvtx(dist.reduce_scatter_fn)(output_tensor,
                                                     input_tensor,
                                                     group=group,
                                                     async_op=False)


@instrument_w_nvtx
@torch.no_grad()
def all_to_all_quant_reduce(tensors: List[Tensor], groups:{}) -> List[Tensor]:

    local_world_size = torch.cuda.device_count()
    global_world_size = dist.get_world_size()
    num_nodes = global_world_size // local_world_size
    #inter_quant_group = 256
    this_rank = dist.get_rank()
    #print(f"num_nodes is {num_nodes}, local_world_size is {local_world_size}, global_world_size is {global_world_size}, this_rank is {this_rank}\n")
    intra_idx = int(this_rank/local_world_size)
    #print(f"all-to-all intra node on local group index {intra_idx}\n")
    inter_idx = this_rank%local_world_size
    #print(f"this_rank is {this_rank}, global group index {inter_idx}\n")
    output_lst: List[Tensor] = [None] * len(tensors)
    for idx, tensor in enumerate(tensors):
        #assert tensor.numel() >= global_world_size*2, '===QG: Tensor too small, must be bigger than total_num_gpus * 2'
        if tensor.dim()==1:
            intra_quant_group = global_world_size
            #assert tensor.numel()%intra_quant_group == 0, '===QG: Tensor cannot be evenly divided.'
        else:
            intra_quant_group = max(tensor.shape[0], tensor.shape[1], global_world_size)
        
        inter_quant_group = intra_quant_group // local_world_size
        #if this_rank==0:
            #print(f"intra group is {intra_quant_group}, inter group is {inter_quant_group}, tensor shape is {tensor.shape}")
        #torch.cuda.synchronize()
        intra_quant_int4, intra_q_scales = quantizer_cuda_module.ds_swizzle_quant(tensor, 4, intra_quant_group, 1, num_nodes, local_world_size)
        #torch.cuda.synchronize()
        #print(f"intra_quiat_int4 shape is {intra_quant_int4.shape}, element size is {intra_quant_int4.element_size()}\n")
        local_output = torch.empty_like(intra_quant_int4)
        scale_output = torch.empty_like(intra_q_scales)
        all_to_all_single(local_output, intra_quant_int4, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
        global_input_tensor, global_scales = quantizer_cuda_module.ds_dequant_reduce_quant_int4(local_output, scale_output, intra_quant_group, inter_quant_group)
        global_output = torch.empty_like(global_input_tensor)
        global_scale_output = torch.empty_like(global_scales)
        all_to_all_single(global_output, global_input_tensor, group=groups[f'global_{inter_idx}'])
        all_to_all_single(global_scale_output, global_scales, group=groups[f'global_{inter_idx}'])
        final_output = quantizer_cuda_module.ds_dequant_int4(global_output, global_scale_output, inter_quant_group)
        output_lst[idx] = (sum(list(final_output.chunk(num_nodes)))/num_nodes).view(-1)
        #if this_rank == 0:
            #print(f"size of outout tensor is {output_lst[idx].shape}, element size is {output_lst[idx].element_size()}\n")
    return output_lst





@instrument_w_nvtx
@torch.no_grad()
def reduce_scatter_coalesced(
    tensors: List[Tensor],
    group: ProcessGroup = None,
) -> List[Tensor]:
    """simultaneously reduce-scatter a list of tensors - this can be done more
    efficiently than individual reduce scatter calls
    TODO. see if PyTorch team wants a c++ version of this for ProcessGroupNCCL
    """
    this_rank = dist.get_rank(group)
    world_sz = dist.get_world_size(group)

    partition_lst_for_each_tensor = [None] * len(tensors)
    for tensor_idx, tensor in enumerate(tensors):
        flattened_tensor = tensor.view(-1)
        chunk_sz = math.ceil(tensor.numel() / world_sz)
        partition_lst_for_each_tensor[tensor_idx] = [
            flattened_tensor[rank * chunk_sz:rank * chunk_sz + chunk_sz]
            for rank in range(0,
                              world_sz)
        ]

    padded_partition_sz_for_each_tensor = tuple(
        math.ceil(t.numel() / world_sz) for t in tensors)

    if len(tensors) == 1 and tensors[0].numel() % world_sz == 0:
        # if there's only one tensor being reduced and we don't need to pad
        # we have an opportunity to avoid a memory allocation
        tensor_partition_flat_buffer = tensors[0].view(-1)
    else:
        # interleave tensor partitions such that the correct reduced partitions of each tensor
        # end up at each rank
        tensor_partitions_lst_with_padding = []
        for rank in range(world_sz):
            for tensor_idx in range(len(tensors)):
                # add tensor content
                tensor_chunk = partition_lst_for_each_tensor[tensor_idx][rank]
                tensor_partitions_lst_with_padding.append(tensor_chunk)

                # add padding if necessary
                padding_sz = padded_partition_sz_for_each_tensor[
                    tensor_idx] - tensor_chunk.numel()
                if padding_sz > 0:
                    tensor_partitions_lst_with_padding.append(
                        torch.empty(padding_sz,
                                    dtype=tensor_chunk.dtype,
                                    device=tensor_chunk.device))

        tensor_partition_flat_buffer = instrument_w_nvtx(
            torch.cat)(tensor_partitions_lst_with_padding)

    tensor_partition_flat_buffer.div_(world_sz)  # pre-divide
    tensor_partition_buffer_for_each_rank: List[Tensor] = torch.chunk(
        tensor_partition_flat_buffer,
        world_sz)

    # batched reduce-scatter call
    _torch_reduce_scatter_fn(tensor_partition_flat_buffer,
                             tensor_partition_buffer_for_each_rank[this_rank],
                             group=group)

    # reverse procedure of the interleaving done previously, done on the
    # result of the batched reduce-scatter
    output_lst: List[Tensor] = [None] * len(tensors)
    offset = 0
    for tensor_idx in range(len(tensors)):
        output_lst[tensor_idx] = tensor_partition_buffer_for_each_rank[this_rank].narrow(
            0,
            offset,
            partition_lst_for_each_tensor[tensor_idx][this_rank].numel())

        offset += padded_partition_sz_for_each_tensor[tensor_idx]
        #if this_rank == 0:
            #print(f"reduce_scatter len output is {len(output_lst)}, idx is {tensor_idx}, vals is {output_lst[tensor_idx].shape}\n")
    return output_lst
