'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import json
import math
import importlib
import torch
from torch import nn
from torch.autograd import Function
import time
from ... import op_builder
import torch.nn as nn
import torch.distributed as dist

# Cuda modules will be imported if needed
inference_cuda_module = None

from .transformer_inference import DeepSpeedInferenceConfig, DeepSpeedSelfAttention, DeepSpeedMLP


class DeepSpeedEncoderFunction(Function):
    @staticmethod
    def forward(ctx,
                input,
                mask,
                attn_weights,
                attn_biases,
                mlp_weights,
                mlp_biases,
                attn_norm,
                input_norm,
                config,
                func,
                norm_factor):
        func(input,
             mask,
             input_norm,
             attn_weights,
             attn_biases,
             attn_norm,
             mlp_weights,
             mlp_biases,
             config.heads,
             config.pre_layer_norm,
             config.epsilon,
             norm_factor,
             config.q_int,
            #  config.bits,
             mlp_weights[1].scale,
             mlp_weights[0].scale,
             attn_weights[1].scale,
             config.enable_qkv_quantization,
             attn_weights[0].scale)
        if config.return_tuple:
            return (input, )
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!')


class DeepSpeedEncoder(nn.Module):
    """Initialize the DeepSpeed Encoder Layer.
        Arguments:
            layer_id: The layer index starting from 0, e.g. if model has 24 transformer layers,
                layer_id will be 0,1,2...23 when each layer object is instantiated
            config: An object of DeepSpeedInferenceConfig
            mp_group: Model parallelism group initialized on the modeling side.
            quantize_scales: This argument groups all the layers' scales used for quantization
            quantize_groups: Number of groups used for quantizing the model
            merge_count: Shows the number of model-parallel checkpoints merged before running inference.
                We use this argument to control the quantization scale for the model parameters if a bigger
                quantize-grouping than 1 is used.
            mlp_extra_grouping: This flag is used to show a 2x higher number of groups used for the MLP part
                of a Transformer layer. We use this feature for quantization to reduce the convergence impact
                for specific downstream tasks.
    """
    layer_id = 0

    def __init__(self,
                 config,
                 mp_group=None,
                 quantize_scales=None,
                 quantize_groups=1,
                 quantize=False,
                 quantize_bits=8,
                 merge_count=1,
                 mlp_extra_grouping=False,
                 qkv_merging=False):
        super(DeepSpeedEncoder, self).__init__()

        self.config = config
        self.config.layer_id = DeepSpeedEncoder.layer_id
        DeepSpeedEncoder.layer_id += 1

        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        # print("DeepSpeed ENCODER config is ", self.config.__dict__)

        self.attention = DeepSpeedSelfAttention(self.config,
                                                mp_group,
                                                quantize_scales,
                                                quantize_groups,
                                                quantize_bits,
                                                merge_count,
                                                qkv_merging)
        self.mlp = DeepSpeedMLP(self.config,
                                mp_group,
                                quantize_scales,
                                quantize_groups,
                                quantize_bits,
                                merge_count,
                                mlp_extra_grouping)

        self.norm_w = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.norm_b = nn.Parameter(torch.Tensor(self.config.hidden_size))
        self.encoder_func = inference_cuda_module.encoder_fp16 if config.fp16 or config.q_int else \
                                    inference_cuda_module.encoder_fp32
        self.attention.norm_factor = (1 / self.attention.norm_factor)**2

    def forward(self,
                input=None,
                input_mask=None,
                attn_mask=None,
                x=None,
                attention_mask=None,
                head_mask=None,
                layer_past=None,
                get_key_value=False,
                get_present=False,
                encoder_output=None,
                enc_dec_attn_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False):

        input_mask = (input_mask if attn_mask is None else
                      attn_mask) if attention_mask is None else attention_mask
        input = x if input is None else input
        return DeepSpeedEncoderFunction.apply(
            input,
            input_mask,
            [self.attention.attn_qkvw,
             self.attention.attn_ow],
            [self.attention.attn_qkvb,
             self.attention.attn_ob],
            [self.mlp.inter_w,
             self.mlp.output_w],
            [self.mlp.inter_b,
             self.mlp.output_b],
            [self.mlp.attn_nw,
             self.mlp.attn_nb],
            [self.norm_w,
             self.norm_b],
            self.config,
            self.encoder_func,
            self.attention.norm_factor)
