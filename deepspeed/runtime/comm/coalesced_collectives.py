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
        #flat_tensor = tensor.view(-1)

        # E3 local-AA-INT4-Comm-only (M+scales)
        '''
        input_tensor = tensor.chunk(2)[0]
        local_output = torch.empty_like(input_tensor)
        scales = torch.rand(tensor.shape[0]).cuda()
        scale_output = torch.empty_like(scales)
        all_to_all_single(local_output, input_tensor, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output, scales, group=groups[f'local_{intra_idx}'])
        '''
        # E3-1 local-AA-INT4-Comm+Kernels
        '''
        if tensor.dim() == 1:
            intra_quant_group = tensor.shape[0] // 64
        else:
            intra_quant_group = max(tensor.shape[0], tensor.shape[1])  

        intra_quant_int4, intra_q_scales = quantizer_cuda_module.ds_act_quant_int4(tensor, intra_quant_group)
        
        local_output = torch.empty_like(intra_quant_int4)
        scale_output = torch.empty_like(intra_q_scales)
        all_to_all_single(local_output, intra_quant_int4, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output, intra_q_scales, group=groups[f'local_{intra_idx}'])
        
        intra_dequant_fp16 = quantizer_cuda_module.ds_dequant_int4(local_output, scale_output, intra_quant_group)        
        output_lst[idx] = (sum(list(intra_dequant_fp16.chunk(local_world_size)))/local_world_size).view(-1) 
        '''
        # E4 global-AA-INT4(M/8)-Comm-only
        '''
        input_tensor = tensor.chunk(16)[0]
        local_output = torch.empty_like(input_tensor)
        scales = torch.rand(tensor.shape[0]).chunk(8)[0].cuda()
        scale_output = torch.empty_like(scales)
        all_to_all_single(local_output, input_tensor, group=groups[f'global_{inter_idx}'])
        all_to_all_single(scale_output, scales, group=groups[f'global_{inter_idx}'])
        '''
        # E4-1 global-AA-INT4(M/8)-Comm+Kernels
        '''
        input_tensor = tensor.chunk(8)[0]
        if tensor.dim() == 1:
            inter_quant_group = input_tensor.shape[0] // 64
        else:
            inter_quant_group = max(input_tensor.shape[0], input_tensor.shape[1])  

        inter_quant_int4, inter_q_scales = quantizer_cuda_module.ds_act_quant_int4(input_tensor, inter_quant_group)
        
        global_output = torch.empty_like(inter_quant_int4)
        scale_output = torch.empty_like(inter_q_scales)
        all_to_all_single(global_output, inter_quant_int4, group=groups[f'global_{inter_idx}'])
        all_to_all_single(scale_output, inter_q_scales, group=groups[f'global_{inter_idx}'])
        
        inter_dequant_fp16 = quantizer_cuda_module.ds_dequant_int4(global_output, scale_output, inter_quant_group)        
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/num_nodes).view(-1)         
        '''
        # E5: E3 followed by E4
        '''
        input_tensor = tensor.chunk(4)[0]
        local_output = torch.empty_like(input_tensor)
        scales = torch.rand(tensor.shape[0]).cuda()
        scale_output = torch.empty_like(scales)
        all_to_all_single(local_output, input_tensor, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output, scales, group=groups[f'local_{intra_idx}'])
        
        global_input = local_output.chunk(8)[0]
        global_scale_input = scale_output.chunk(8)[0]
        global_output = torch.empty_like(global_input)
        global_scale = torch.empty_like(global_scale_input)
        all_to_all_single(global_output, global_input, group=groups[f'global_{inter_idx}'])
        all_to_all_single(global_scale, global_scale_input, group=groups[f'global_{inter_idx}'])        
        '''

        # E6: 2x E3(M/2)
        '''
        input_tensor1 = tensor.chunk(8)[0]
        local_output1 = torch.empty_like(input_tensor1)
        scales1 = torch.rand(tensor.shape[0]).cuda()
        scale_output1 = torch.empty_like(scales1)
        all_to_all_single(local_output1, input_tensor1, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output1, scales1, group=groups[f'local_{intra_idx}'])

        input_tensor2 = tensor.chunk(8)[1]
        local_output2 = torch.empty_like(input_tensor2)
        scales2 = torch.rand(tensor.shape[0]).cuda()
        scale_output2 = torch.empty_like(scales2)
        all_to_all_single(local_output2, input_tensor2, group=groups[f'local_{intra_idx}'])
        all_to_all_single(scale_output2, scales2, group=groups[f'local_{intra_idx}'])
        '''

        # E7: 2x E4(M/16)
        '''
        input_tensor1 = tensor.chunk(64)[0]
        local_output1 = torch.empty_like(input_tensor1)
        scales1 = torch.rand(tensor.shape[0]).chunk(16)[0].cuda()
        scale_output1 = torch.empty_like(scales1)
        all_to_all_single(local_output1, input_tensor1, group=groups[f'global_{inter_idx}'])
        all_to_all_single(scale_output1, scales1, group=groups[f'global_{inter_idx}'])

        input_tensor2 = tensor.chunk(64)[1]
        local_output2 = torch.empty_like(input_tensor2)
        scales2 = torch.rand(tensor.shape[0]).chunk(16)[1].cuda()
        scale_output2 = torch.empty_like(scales2)
        all_to_all_single(local_output2, input_tensor2, group=groups[f'global_{inter_idx}'])
        all_to_all_single(scale_output2, scales2, group=groups[f'global_{inter_idx}'])
        '''

        # E8: Pipeline E6, E7
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        event_1 = torch.cuda.Event(False, False, False)
        event_2 = torch.cuda.Event(False, False, False)
        #tensor = tensor.t()
        intra_quant_group = max(tensor.shape[0], tensor.shape[1])
        if tensor.dim()==1:
            intra_quant_group = 64
        inter_quant_group = intra_quant_group // 8
        intra_quant_int4, intra_q_scales = quantizer_cuda_module.ds_act_quant_int4(tensor, intra_quant_group)
        input_tensor1, input_tensor2 = intra_quant_int4.chunk(2)
        scales1, scales2= intra_q_scales.chunk(2)
        #if this_rank==0:
            #print(f"intput_tensor1 is {input_tensor1}\n")
        local_output1 = torch.empty_like(input_tensor1)
        scale_output1 = torch.empty_like(scales1)
        local_output2 = torch.empty_like(input_tensor2)
        scale_output2 = torch.empty_like(scales2)

        #torch.cuda.synchronize()
        #dist.barrier()
        #quant_2 = time.time()
        #print(f"Guanhua init quant for all-to-all is: {quant_2 - quant_1}\n")
        # Intra comm on S1, records two different events
        with torch.cuda.stream(s1):
            all_to_all_single(local_output1, input_tensor1, group=groups[f'local_{intra_idx}'], async_op=True)
            all_to_all_single(scale_output1, scales1, group=groups[f'local_{intra_idx}'], async_op=True)
        s1.record_event(event_1)
        with torch.cuda.stream(s1):
            all_to_all_single(local_output2, input_tensor2, group=groups[f'local_{intra_idx}'], async_op=True)
            all_to_all_single(scale_output2, scales2, group=groups[f'local_{intra_idx}'], async_op=True)
        s1.record_event(event_2)


        # Inter comm on S2, wait for two different events
        s2.wait_event(event_1)
        #torch.cuda.synchronize()
        #dist.barrier()
        #intra_1 = time.time()
        #print(f"Guanhua 1-half intra all-to-all is: {intra_1 - quant_2}\n")
        intra_dequant1 = quantizer_cuda_module.ds_dequant_int4(local_output1, scale_output1, intra_quant_group//2)
        intra_reduce1 = sum(list(intra_dequant1.chunk(local_world_size)))

        global_input_tensor1, global_scales1 = quantizer_cuda_module.ds_act_quant_int4(intra_reduce1, inter_quant_group//2)
        #if this_rank==0:
            #print(f"global_intput_tensor1 shape is {global_input_tensor1}\n")
        global_output1 = torch.empty_like(global_input_tensor1)
        global_scale_output1 = torch.empty_like(global_scales1)
        with torch.cuda.stream(s2):
            all_to_all_single(global_output1, global_input_tensor1, group=groups[f'global_{inter_idx}'], async_op=True)
            all_to_all_single(global_scale_output1, global_scales1, group=groups[f'global_{inter_idx}'], async_op=True)
        #s2.record_event(event_3)

        s2.wait_event(event_2)
        intra_dequant2 = quantizer_cuda_module.ds_dequant_int4(local_output2, scale_output2, intra_quant_group//2)
        intra_reduce2 = sum(list(intra_dequant2.chunk(local_world_size)))

        global_input_tensor2, global_scales2 = quantizer_cuda_module.ds_act_quant_int4(intra_reduce2, inter_quant_group//2)
        global_output2 = torch.empty_like(global_input_tensor2)
        global_scale_output2 = torch.empty_like(global_scales2)

        with torch.cuda.stream(s2):
            all_to_all_single(global_output2, global_input_tensor2, group=groups[f'global_{inter_idx}'], async_op=True)
            all_to_all_single(global_scale_output2, global_scales2, group=groups[f'global_{inter_idx}'], async_op=True)
        #s2.record_event(event_4)
        #torch.cuda.synchronize()
        #dist.barrier()
        #dequant_1 = time.time()
        #print(f"Guanhua inter all-to-all is: {dequant_1 - intra_1}\n")
        s2.synchronize()
        inter_output_single = torch.concat((global_output1, global_output2))
        inter_q_scale_out = torch.concat((global_scale_output1, global_scale_output2))
        inter_dequant_fp16 = quantizer_cuda_module.ds_dequant_int4(inter_output_single, inter_q_scale_out, inter_quant_group)#.t()
        #if this_rank == 0:
        #print(f"inter_dequant_fp16 is {inter_dequant_fp16}\n")
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/global_world_size).view(-1)
        #torch.cuda.synchronize()
        #dist.barrier()
        #dequant_2 = time.time()
        #print(f"Guanhua final dequant for all-to-all is: {dequant_2 - dequant_1}\n")
        
        #if this_rank == 0:
            #print(f"output idx is {idx}, shape is {output_lst[idx].shape}\n")


        # E9: Pipeline Reduce-scatter, E7
        '''
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        event_1 = torch.cuda.Event(False, False, False)
        event_2 = torch.cuda.Event(False, False, False)
        tensor1, tensor2 = tensor.chunk(2)
        local_output1 = torch.empty_like(tensor1.chunk(8)[0])
        local_output2 = torch.empty_like(local_output1)

        #torch.cuda.synchronize()
        #dist.barrier()
        #quant_2 = time.time()
        #print(f"Guanhua init quant for all-to-all is: {quant_2 - quant_1}\n")
        # Intra comm on S1, records two different events
        with torch.cuda.stream(s1):
            _torch_reduce_scatter_fn(tensor1, local_output1, group=groups[f'local_{intra_idx}'], async_op=True)
        s1.record_event(event_1)
        with torch.cuda.stream(s1):
            _torch_reduce_scatter_fn(tensor2, local_output2, group=groups[f'local_{intra_idx}'], async_op=True)
        s1.record_event(event_2)


        # Inter comm on S2, wait for two different events
        s2.wait_event(event_1)
        global_input_tensor1, global_scales1 = quantizer_cuda_module.gh_quant_fp16(local_output1, inter_quant_group//2)
        #if this_rank==0:
            #print(f"global_intput_tensor1 shape is {global_input_tensor1}\n")
        global_output1 = torch.empty_like(global_input_tensor1)
        global_scale_output1 = torch.empty_like(global_scales1)
        with torch.cuda.stream(s2):
            all_to_all_single(global_output1, global_input_tensor1, group=groups[f'global_{inter_idx}'], async_op=True)
            all_to_all_single(global_scale_output1, global_scales1, group=groups[f'global_{inter_idx}'], async_op=True)

        s2.wait_event(event_2)
        global_input_tensor2, global_scales2 = quantizer_cuda_module.gh_quant_fp16(local_output2, inter_quant_group//2)
        global_output2 = torch.empty_like(global_input_tensor2)
        global_scale_output2 = torch.empty_like(global_scales2)

        with torch.cuda.stream(s2):
            all_to_all_single(global_output2, global_input_tensor2, group=groups[f'global_{inter_idx}'], async_op=True)
            all_to_all_single(global_scale_output2, global_scales2, group=groups[f'global_{inter_idx}'], async_op=True)
        
        s2.synchronize()
        #torch.cuda.synchronize()
        #dist.barrier()
        #dequant_1 = time.time()
        #print(f"Guanhua inter all-to-all is: {dequant_1 - intra_1}\n")
        inter_output_single = torch.concat((global_output1, global_output2))
        inter_q_scale_out = torch.concat((global_scale_output1, global_scale_output2))
        inter_dequant_fp16 = quantizer_cuda_module.gh_dequant_fp16(inter_output_single, inter_q_scale_out, inter_quant_group)
        #if this_rank == 0:
        #print(f"inter_dequant_fp16 is {inter_dequant_fp16}\n")
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/num_nodes).view(-1)
        '''
        '''
        # Intra-machine all-to-all
        #if this_rank == 0:
            #print(f"tensors len is {len(tensors)}, tensor shape is {tensor.shape}\n")
        input_tensor = list(tensor.chunk(local_world_size))
        local_output = torch.empty_like(tensor.chunk(local_world_size)[0])
        #reduce_scatter(local_output, input_tensor, group=groups[f'local_{intra_idx}'])
        _torch_reduce_scatter_fn(tensor, local_output, group=groups[f'local_{intra_idx}'])
        #local_output1 = local_output/local_world_size
        # Inter-machine all-to-all
        #local_output = local_output.chunk(2)[0]     
        #inter_quant_int8, inter_q_scales = quantizer_cuda_module.gh_quant_fp16(local_output, inter_quant_group)
        #if local_output.dim()==1:
            #local_output = local_output.view(-1,32)
        #print(f"local output is {local_output}\n")
        if local_output.dim() == 1:
            inter_quant_group = local_output.shape[0] // 64
        else:
            inter_quant_group = max(local_output.shape[0], local_output.shape[1])
        inter_quant_int8, inter_q_scales = quantizer_cuda_module.ds_act_quant_int4(local_output, inter_quant_group)
        #print(f"inter_quant_int8 is {inter_quant_int8}, inter_q_scales is {inter_q_scales}\n")
        inter_output_single = torch.empty_like(inter_quant_int8)
        inter_q_scale_out = torch.empty_like(inter_q_scales)

        all_to_all_single(inter_output_single, inter_quant_int8, group=groups[f'global_{inter_idx}'])
        all_to_all_single(inter_q_scale_out, inter_q_scales, group = groups[f'global_{inter_idx}'])
        #torch.cuda.synchronize()
        inter_dequant_fp16 = quantizer_cuda_module.ds_dequant_int4(inter_output_single, inter_q_scale_out, inter_quant_group)
        #if this_rank == 0:
        #print(f"inter_dequant_fp16 is {inter_dequant_fp16}\n")
        output_lst[idx] = (sum(list(inter_dequant_fp16.chunk(num_nodes)))/global_world_size).view(-1)
        
        #torch.cuda.synchronize()
        #if this_rank == 0:
            #print(f"local_output shape is {local_output.shape}, all_to_all len output is {len(output_lst)}, idx is {idx}, shape is {output_lst[idx].shape}, vals is {output_lst[idx]}\n")
        '''
    return None






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

    #if this_rank == 0:
    #print(f"===reduce_scatter output is {output_lst}, len reduce_scatter output is {len(output_lst)}\n")
    #return output_lst