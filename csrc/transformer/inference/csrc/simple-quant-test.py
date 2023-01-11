import torch

from deepspeed.ops import op_builder

quantizer_cuda_module = op_builder.InferenceBuilder().load()

rows = 4096
cols = 4096
# Number of groups
groups = 512
gpu_per_node = 16

input_tensor = torch.randn(rows, cols, dtype = torch.float16).cuda()
#input_tensor = input_tensor.view(-1)
print(f"input tensor is {input_tensor}\n")
test_tensor = input_tensor.clone().detach()

quant_int4, scales_int4 = quantizer_cuda_module.ds_swizzle_quant(test_tensor, 4, groups, 1, 1, gpu_per_node)
print(f"quant_int4 is {quant_int4}, scales_int4 is {scales_int4}\n")
global_input_tensor, global_scales = quantizer_cuda_module.ds_dequant_reduce_quant_int4(quant_int4, scales_int4, groups, groups//gpu_per_node)
dequant_int4 = quantizer_cuda_module.ds_dequant_int4(global_input_tensor, global_scales, groups//gpu_per_node)
print(f"dequant_int4 is {dequant_int4}\n")