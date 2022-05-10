import torch
import os
from torch.utils.cpp_extension import load

CUDA_LIB64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
CUDA_INCLUDE = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")

op_module = load(name='bg',
                 sources=['test.cpp'],
                 extra_include_paths=['./', CUDA_INCLUDE],
                 extra_cflags=[
                     '-O3',
                     '-std=c++14',
                     f'-L{CUDA_LIB64}',
                     '-lcudart',
                     '-lcublas',
                     '-Wno-reorder',
                 ],
                 extra_ldflags=[
                     '-L/home/reyazda/gpt3-int8/deepspeed/ops/libs/cutlass',
                     '-lgemmlib',
                 ],
                 verbose=True)

import deepspeed

from deepspeed.ops import op_builder

q_module = op_builder.QuantizerBuilder().load()
i_module = op_builder.InferenceSpecializedBuilder().load()
def quantize(inputs, bit, quantize_group='per_tensor', symmetric=False, stochastic_round=False, num_groups=1, do_scale=False):

    q_range = 2**bit
    input_flat = inputs.float().reshape(num_groups, -1).contiguous()

    input_flat = torch.nan_to_num(input_flat, nan=0.0)

    if stochastic_round:
        noise = input_flat.new(inputs.shape).uniform_(-0.5, 0.5)
        mask_init = inputs.amax(-1, keepdim=True) - inputs.amin(-1, keepdim=True)
        if (mask_init.abs() == 0).sum():
            zero_mask = mask_init.new_ones(mask_init.shape)
            zero_mask[mask_init.abs() == 0] = 0.
            noise *= zero_mask
        noise = noise.reshape(input_flat.size()).contiguous()
    else:
        noise = 0

    input_min = input_flat.amin(-1, keepdim=True)
    input_max = input_flat.amax(-1, keepdim=True)

    if symmetric:
        # scale = torch.max(input_min.abs(), input_max.abs()) * 2.0 / q_range
        scale = q_range /  (2 * torch.max(input_min.abs(), input_max.abs()) + 1e-5)
    else:
        scale = (input_max - input_min) / q_range
        zero_point = (input_min / scale).round() * scale # make sure the zero point is mapped to an interger value

    scale[scale.abs() == 0.] = 1.

    if symmetric:
        input_flat = (input_flat * scale + noise).round().clamp(-q_range // 2, q_range // 2 - 1)
        input_flat_scaled = input_flat / (scale if do_scale else 1.0)
    else:
        input_flat = ( (input_flat - zero_point ) / scale + noise ).round().clamp(0, (q_range - 1)) * scale + zero_point

    return input_flat.reshape(inputs.shape).to(torch.int8).contiguous(), 1/scale.view(-1).contiguous(), input_flat_scaled.reshape(inputs.size()).contiguous()
for _ in range(1):
    aa = torch.randn(3, 768).cuda().half() / 10
    b = torch.randn(768,768).cuda().half() / 100
    num_groups = 32

    w_int, w_s = q_module.ds_quantizer(b.t().contiguous(), 8, num_groups)
    a_int, a_s = q_module.ds_quantizer(aa, 8, aa.size(0))
    ww = w_int.reshape(num_groups, -1).contiguous()

    w_q = (ww * w_s.unsqueeze(1)).reshape(w_int.shape).contiguous()
    w_q = w_q.t().contiguous()
    w_int = w_int.t().contiguous()
    a_q = a_int * a_s.unsqueeze(1)
    a_q = a_int.float().contiguous()

    weight_int, weight_scale, weight_q = quantize(b.t().contiguous(), 8, quantize_group='weight', num_groups=num_groups, symmetric=True, do_scale=True)
    weight_int = weight_int.t().contiguous()
    weight_q = weight_q.t().contiguous()
    act_int, act_scale, act_q = quantize(aa, 8, quantize_group='weight', num_groups=aa.size(0), symmetric=True)
#    import pdb;pdb.set_trace()
    #for i in range(40):
    #    print(f"---------------- {i}: {torch.allclose(w_q[i], weight_q[i])} , {((weight_q[i].float() - w_q[i].float()).abs()/(weight_q[i].abs()+1e-6)).sum() / weight_q[i].numel()} ---------------")
    #    if ((weight_q[i].float() - w_q[i].float()).abs()/(weight_q[i].abs()+1e-6)).sum() / weight_q[i].numel() > 0.01:
    #        for k in range(w_q.shape[1]):
    #            if (w_q[i][k] - weight_q[i][k]).abs() / (weight_q[i][k].abs()+1e-6) > 0.01:
    #                print(weight_q[i][k].item(), w_q[i][k].item(), w_int[i][k].item(), weight_int[i][k].item(), weight_scale[i].item(), w_s[i].item())
    #    #print(weight_q[i].norm())
    #    #print(w_q[i].norm())
    print(((weight_q.float() - w_q.float()).abs()/(weight_q.abs()+1e-6)).sum() / weight_q.numel())
    print(((act_q.float() - a_q.narrow(0,0,aa.size(0)).float()).abs()/(act_q.abs()+1e-6)).sum() / act_q.numel())

    #print(act_q)
    #print(a_q)
    #exit(1)
        #print("-------------------------------")

    #aa = aa.to(torch.int8)
    #b = b.to(torch.int8)
    #alpha = torch.range(1,64,1, device='cuda')
    alpha = torch.ones(128, device='cuda') #torch.range(1, a.size(0), 1, device='cuda')
    alpha1 = torch.ones(40, device='cuda')
    #alpha1[20:] *= 2
    #alpha[aa.size(0)//2:] *= 2
    beta = torch.zeros(1, dtype=torch.half, device='cuda')
    import time
    torch.cuda.synchronize()
    t0 = time.time()
    b_t = weight_int.transpose(-1,-2).contiguous()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1):
#        ds_result = i_module.run_q_gemm(aa, act_scale, b_t, weight_scale)
#        import pdb;pdb.set_trace()
        ds_result = op_module.cutlass_gemm(a_int, b_t, a_s, weight_scale, beta)

    print(ds_result.shape)
    ds_result = ds_result.narrow(0,0,aa.size(0))
    print(ds_result)
    print(ds_result.shape)
    torch.cuda.synchronize()
    #aa = aa.half()
    #bb = b.half()
    torch.cuda.synchronize()
    t0 = time.time()
    #print("torch scale is ", (act_scale*weight_scale).item())
    pt_result = (torch.matmul(act_q,weight_q).reshape(aa.size(0),-1)*act_scale.unsqueeze(1)).half()
    #pt_result[pt_result.size(0)//2:, :] *= 2

    #pt_result[:, pt_result.size(1)//2:] *= 2
    print(pt_result)
    print(pt_result.shape)
    print(((pt_result.float()-ds_result.float()).abs() / (1e-6+pt_result.float().abs())).sum() / ds_result.numel())

#for i in range(128):
#    print(((pt_result[i].float()-ds_result[i].float()).abs() / (1e-6+pt_result[i].float().abs())).sum() / ds_result[i].numel())
        #for k in range(ds_result.size(1)):
        #    if(((pt_result[i][k].float()-ds_result[i][k].float()).abs() / (1e-6+pt_result[i][k].float().abs())) > 0.1):
        #        print(pt_result[i][k].item(), ds_result[i][k].item(), ((pt_result[i][k].float()-ds_result[i][k].float()).abs() / (1e-6+pt_result[i][k].float().abs())))

torch.cuda.synchronize()
#print(A)
#print(B)
#print(f'error is: {((A-B).float()).abs().sum() / A.numel()}')
