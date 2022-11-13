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

'''
@instrument_w_nvtx
@torch.no_grad()
def all_to_all_quant_reduce(tensors: List[Tensor], groups:{}) -> List[Tensor]:
    #torch.set_printoptions(threshold=40960)
    local_world_size = torch.cuda.device_count()
    global_world_size = dist.get_world_size()
    num_nodes = global_world_size // local_world_size
    this_rank = dist.get_rank()
    #print(f"num_nodes is {num_nodes}, local_world_size is {local_world_size}, global_world_size is {global_world_size}, this_rank is {this_rank}\n")
    intra_idx = int(this_rank/local_world_size)
    #print(f"all-to-all intra node on local group index {intra_idx}\n")
    inter_idx = this_rank%local_world_size
    #print(f"this_rank is {this_rank}, global group index {inter_idx}\n")
    output_lst: List[Tensor] = [None] * len(tensors)
    for idx, tensor in enumerate(tensors):
        event_1 = torch.cuda.Event()
        event_2 = torch.cuda.Event()
        event_3 = torch.cuda.Event()
        event_4 = torch.cuda.Event()
        assert tensor.numel()>=256, 'Tensor too small, must be bigger than 256'
        if tensor.dim()==1:
            #assert tensor.shape[0]>=256, '1dim num vals must bigger than 256'
            intra_quant_group = 256
        else:
            #assert tensor.shape[0]*tensor.shape[1]>=256, '2dim num vals must bigger than 256'
            intra_quant_group = max(tensor.shape[0], tensor.shape[1], 256)
        inter_quant_group = intra_quant_group // 8

        intra_quant_int4, intra_q_scales = quantizer_cuda_module.ds_act_quant_int4(tensor, intra_quant_group)
        input_tensor1, input_tensor2 = intra_quant_int4.chunk(2)
        scales1, scales2= intra_q_scales.chunk(2)
        #local_output = torch.empty_like(intra_quant_int4)
        #scale_output = torch.empty_like(intra_q_scales)
        local_output1 = torch.zeros_like(input_tensor1)
        scale_output1 = torch.zeros_like(scales1)
        local_output2 = torch.zeros_like(input_tensor2)
        scale_output2 = torch.zeros_like(scales2)
        #for i in range(8):
            #if this_rank == i:
                #print(f"intra_quant_int4 is {intra_quant_int4}, shape is {intra_quant_int4.shape}\n")
                #print(f"intra_q_scales is {intra_q_scales}, shape is {intra_q_scales.shape}\n")
        s1=torch.cuda.current_stream()
        with torch.cuda.stream(s1):
            all_to_all_single(local_output1, input_tensor1, group=groups[f'local_{intra_idx}'])
            all_to_all_single(scale_output1, scales1, group=groups[f'local_{intra_idx}'])
            #event_1.record()
            s1.record_event(event_1)
            all_to_all_single(local_output2, input_tensor2, group=groups[f'local_{intra_idx}'])
            all_to_all_single(scale_output2, scales2, group=groups[f'local_{intra_idx}'])
            #event_2.record()
            s1.record_event(event_2)

        s2 = torch.cuda.Stream()
        s2.wait_event(event_1)
        # Inter comm on S2, wait for two different events
        #global_input_tensor1, global_scales1 = quantizer_cuda_module.ds_dequant_reduce_quant_int4(local_output1, scale_output1, intra_quant_group//2, inter_quant_group//2)
        #for i in range(8):
            #if this_rank == i:
                #print(f"local_output1 is {local_output1}, shape is {local_output1.shape}\n")
                #print(f"scale_output1 is {scale_output1}, shape is {scale_output1.shape}\n")
        global_input_tensor1, global_scales1 = quantizer_cuda_module.ds_dequant_reduce_quant_int4(local_output1, scale_output1, intra_quant_group//2, inter_quant_group//2)
        #for j in range(16):
            #for i in range(8):
                #if this_rank == i+j*8:
                    #print(f"global_input1 is {global_input_tensor1}, shape is {global_input_tensor1.shape}\n")
                    #print(f"global_input_scales1 is {global_scales1}, shape is {global_scales1.shape}\n")        
        global_output1 = torch.empty_like(global_input_tensor1)
        global_scale_output1 = torch.empty_like(global_scales1)
        with torch.cuda.stream(s2):
            all_to_all_single(global_output1, global_input_tensor1, group=groups[f'global_{inter_idx}'])
            all_to_all_single(global_scale_output1, global_scales1, group=groups[f'global_{inter_idx}'])
        s2.record_event(event_3)
        
        s2.wait_event(event_2)
        #s2.synchronize()
        #for i in range(8):
            #if this_rank == i:
                #print(f"local_output2 is {local_output2}, shape is {local_output2.shape}\n")
                #print(f"scale_output2 is {scale_output2}, shape is {scale_output2.shape}\n")
        global_input_tensor2, global_scales2 = quantizer_cuda_module.ds_dequant_reduce_quant_int4(local_output2, scale_output2, intra_quant_group//2, inter_quant_group//2)
        #for j in range(16):
            #for i in range(8):
                #if this_rank == i+j*8:
                    #print(f"global_intput2 is {global_input_tensor2}, shape is {global_input_tensor2.shape}\n")
                    #print(f"global_intput_scales2 is {global_scales2}, shape is {global_scales2.shape}\n")

        global_output2 = torch.empty_like(global_input_tensor2)
        global_scale_output2 = torch.empty_like(global_scales2)

        with torch.cuda.stream(s2):
            all_to_all_single(global_output2, global_input_tensor2, group=groups[f'global_{inter_idx}'])
            all_to_all_single(global_scale_output2, global_scales2, group=groups[f'global_{inter_idx}'])
        s2.record_event(event_4)
        s2.synchronize()
        
        s1 = torch.cuda.current_stream()
        s1.wait_event(event_3)
        #event_3.wait()
        #for j in range(16):
            #for i in range(8):
                #if this_rank == i+j*8:
                    #print(f"global_output1 is {global_output1}, shape is {global_output1.shape}\n")
                    #print(f"global_scale_output1 is {global_scale_output1}, shape is {global_scale_output1.shape}\n")
        final_output1 = quantizer_cuda_module.ds_dequant_int4(global_output1, global_scale_output1, inter_quant_group//2)
        #for j in range(16):
            #for i in range(8):
                #if this_rank == i+j*8:
                    #print(f"final_output1  is {final_output1 }, shape is {final_output1.shape}\n")
        s1.wait_event(event_4)
        #event_4.wait()
        s1.synchronize()
        #if this_rank == 0:
            #print(f"global_output2 is {global_output2}, shape is {global_output2.shape}\n")
            #print(f"global_scale_output2 is {global_scale_output2}, shape is {global_scale_output2.shape}\n")
        final_output2 = quantizer_cuda_module.ds_dequant_int4(global_output2, global_scale_output2, inter_quant_group//2)
        inter_dequant_fp16 = torch.concat((final_output1, final_output2))
        #print(f"inter_dequant_fp16 is {inter_dequant_fp16}\n")
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/num_nodes).view(-1)
        #temp = torch.randn_like(output_lst[idx])
        #for i in range(8):
            #if this_rank == i:
                #print(f"output tensor is {output_lst[idx]}, shape is {output_lst[idx].shape}\n")
        #torch.cuda.synchronize()
    return output_lst[idx]

'''

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

        local_output = torch.empty_like(tensor.chunk(local_world_size)[0])
        #reduce_scatter(local_output, input_tensor, group=groups[f'local_{intra_idx}'])
        _torch_reduce_scatter_fn(tensor, local_output, group=groups[f'local_{intra_idx}'])

        if local_output.dim() == 1:
            inter_quant_group = local_output.shape[0] // 32
        else:
            inter_quant_group = max(local_output.shape[0], local_output.shape[1],256)
        inter_quant_int8, inter_q_scales = quantizer_cuda_module.ds_act_quant_int4(local_output, inter_quant_group)
        inter_output_single = torch.empty_like(inter_quant_int8)
        inter_q_scale_out = torch.empty_like(inter_q_scales)

        all_to_all_single(inter_output_single, inter_quant_int8, group=groups[f'global_{inter_idx}'])
        all_to_all_single(inter_q_scale_out, inter_q_scales, group = groups[f'global_{inter_idx}'])
        #torch.cuda.synchronize()
        inter_dequant_fp16 = quantizer_cuda_module.ds_dequant_int4(inter_output_single, inter_q_scale_out, inter_quant_group)

        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/global_world_size).view(-1)
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
            #print(f"reduce_scatter output is {output_lst[tensor_idx]}, idx is {tensor_idx}, vals is {output_lst[tensor_idx].shape}\n")
    
    #output_lst: List[Tensor] = [None] * len(tensors)
    #for idx, tensor in enumerate(tensors):
        #output_lst[idx] = tensor.chunk(128)[0].view(-1) 
    return output_lst
