import torch

from deepspeed.ops import op_builder

quantizer_cuda_module = op_builder.InferenceBuilder().load()

rows = 8
cols = 12 
# Number of groups
groups = 4

input_tensor = torch.randn(rows, cols, dtype = torch.float16).cuda()
input_tensor = input_tensor.view(-1)
print(f"input tensor is {input_tensor}\n")
test_tensor = input_tensor.clone().detach()

quant_int4, scales_int4 = quantizer_cuda_module.ds_act_quant_int4(input_tensor, groups)
print(f"quant_int4 is {quant_int4}, scales_int4 is {scales_int4}\n")
dequant_int4 = quantizer_cuda_module.ds_dequant_int4(quant_int4, scales_int4, groups)
print(f"dequant_int4 is {dequant_int4}\n")