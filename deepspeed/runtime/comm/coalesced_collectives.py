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
    #print(f"===start all to all ===\n")
    #group = groups['local_0']

    #this_rank = dist.get_rank(group)
    #world_sz = dist.get_world_size(group)
    #print(f"this rank is {this_rank}, world_size is {world_sz}\n")
    #world_sz=8

    # coalesced tensor
    '''
    partition_lst_for_each_tensor = [None] * len(tensors)
    for idx, tensor in enumerate(tensors):
        print(f"===Guanhua print tensor as idx is {idx}, tensor is {tensor}\n")
        flatted_tensor = tensor.view(-1)
        chunk_sz = math.ceil(tensor.numel() / world_sz)
        partition_lst_for_each_tensor[idx] = [flatted_tensor[rank*chunk_sz:(rank+1)*chunk_sz] 
                                                for rank in range(0, world_sz)]
    
    padded_sz_for_each_tensor = tuple(math.ceil(tensor.numel()/ world_sz) for tensor in tensors)

    if len(tensors) == 1 and tensors[0].numel() % world_sz == 0:
        tensor_partition_flat_buffer = tensors[0].view(-1)
    else:
        tensor_partitions_lst = []
        for rank in range(world_sz):
            for tensor_idx in range(len(tensors)):
                tensor_chunk = partition_lst_for_each_tensor[tensor_idx][rank]
                #print(f"tensor_chunk is {tensor_chunk}")
                tensor_partitions_lst.append(tensor_chunk)
        tensor_partition_flat_buffer = torch.cat(tensor_partitions_lst)
    tensor_partition_flat_buffer1 = tensor_partition_flat_buffer
    print(f"flat tensor is {tensor_partition_flat_buffer1}\n")
    '''

    local_world_size = torch.cuda.device_count()
    global_world_size = dist.get_world_size()
    num_nodes = int(global_world_size / local_world_size)
    intra_quant_group = 128
    inter_quant_group = 128
    this_rank = dist.get_rank()
    #print(f"num_nodes is {num_nodes}, local_world_size is {local_world_size}, global_world_size is {global_world_size}, this_rank is {this_rank}\n")
    intra_idx = int(this_rank/local_world_size)
    #print(f"all-to-all intra node on local group index {intra_idx}\n")
    inter_idx = this_rank%local_world_size
    #print(f"this_rank is {this_rank}, global group index {inter_idx}\n")
    output_lst: List[Tensor] = [None] * len(tensors)
    for idx, tensor in enumerate(tensors):

        # Intra-machine all-to-all
        #if this_rank == 0:
            #print(f"tensor idx is {idx}, tensor shape is {tensor.shape}")
        #input_tensor = list(tensor.chunk(local_world_size))
        local_output = torch.empty_like(tensor.chunk(local_world_size)[0])
        #reduce_scatter(local_output, input_tensor, group=groups[f'local_{intra_idx}'])
        _torch_reduce_scatter_fn(tensor, local_output, group=groups[f'local_{intra_idx}'])

        # Inter-machine all-to-all
        #local_output = local_output.chunk(2)[0]     
        #inter_quant_int8, inter_q_scales = quantizer_cuda_module.gh_quant_fp16(local_output, inter_quant_group)
        inter_quant_int8, inter_q_scales = quantizer_cuda_module.ds_act_quant_int4(local_output, inter_quant_group)
        inter_output_single = torch.empty_like(inter_quant_int8)
        inter_q_scale_out = torch.empty_like(inter_q_scales)

        all_to_all_single(inter_output_single, inter_quant_int8, group=groups[f'global_{inter_idx}'])
        all_to_all_single(inter_q_scale_out, inter_q_scales, group = groups[f'global_{inter_idx}'])

        #inter_dequant_fp16 = quantizer_cuda_module.gh_dequant_fp16(inter_output_single, inter_q_scale_out, inter_quant_group)
        inter_dequant_fp16 = quantizer_cuda_module.ds_dequant(inter_output_single, inter_q_scale_out, inter_quant_group)
        #output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/num_nodes).view(-1)
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes//2)))/(num_nodes//2)).view(-1)
        #torch.cuda.synchronize()
        #if this_rank == 0:
            #print(f"all_to_all len output is {len(output_lst)}, idx is {idx}, shape is {output_lst[idx].shape}\n")        
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
            #print(f"reduce_scatter len output is {len(output_lst)}, idx is {tensor_idx}, shape is {output_lst[tensor_idx].shape}\n")
    return output_lst

    #if this_rank == 0:
    #print(f"===reduce_scatter output is {output_lst}, len reduce_scatter output is {len(output_lst)}\n")
    #return output_lst
