'''
Copyright 2020 The Microsoft DeepSpeed Team
'''
import torch
from torch import nn
from torch.autograd import Function
from ... import op_builder

# Cuda modules will be imported if needed
inference_cuda_module = None

from .transformer_inference import DeepSpeedSelfAttention, DeepSpeedMLP


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
                qkv_func,
                mlp_func,
                norm_factor,
                flash_attn_func):
        if flash_attn_func is not None and input.shape[-2] % 128 == 0:
            (qkv,
             ) = qkv_func(input,
                          input_norm,
                          attn_weights[0],
                          attn_biases[0],
                          config.heads,
                          config.pre_layer_norm,
                          config.enable_qkv_quantization,
                          attn_weights[0].scale,
                          config.epsilon,
                          config.layer_id)
            BATCH = input.size(0)
            N_CTX = input.size(1)

            context_layer = flash_attn_func(
                qkv,
                mask,
                torch.empty(1),
                N_CTX,
                0.0, #  dropout
                return_attn_probs=False,
                causal=False
            )
            mlp_func(input,
                     context_layer.view(BATCH,
                                        N_CTX,
                                        input.shape[-1]),
                     attn_weights[1],
                     attn_biases[1],
                     attn_norm,
                     mlp_weights,
                     mlp_biases,
                     input_norm,
                     config.heads,
                     config.pre_layer_norm,
                     config.epsilon,
                     config.q_int8,
                     config.enable_qkv_quantization,
                     mlp_weights[1].scale,
                     mlp_weights[0].scale,
                     attn_weights[1].scale)
        else:
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
                 config.q_int8,
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

        if self.config.layer_id == 0:
            print("DeepSpeed ENCODER config is ", self.config.__dict__)

        self.attention = DeepSpeedSelfAttention(self.config,
                                                mp_group,
                                                quantize_scales,
                                                quantize_groups,
                                                merge_count,
                                                qkv_merging)
        self.mlp = DeepSpeedMLP(self.config,
                                mp_group,
                                quantize_scales,
                                quantize_groups,
                                merge_count,
                                mlp_extra_grouping)

        data_type = torch.half if config.fp16 else torch.float
        device = torch.cuda.current_device() if config.bigscience_bloom else 'cpu'
        self.norm_w = nn.Parameter(
            torch.empty(self.config.hidden_size,
                        dtype=data_type,
                        device=device))
        self.norm_b = nn.Parameter(
            torch.empty(self.config.hidden_size,
                        dtype=data_type,
                        device=device))
        self.encoder_func = inference_cuda_module.encoder_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.encoder_fp32
        self.attention.norm_factor = (1 / self.attention.norm_factor)**2
        self.qkv_func = inference_cuda_module.encoder_qkv_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.encoder_qkv_fp32
        self.mlp_func = inference_cuda_module.encoder_mlp_fp16 if config.fp16 or config.q_int8 else \
                                    inference_cuda_module.encoder_mlp_fp32

        try:
            from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
            self.flash_attn_func = flash_attn_unpadded_qkvpacked_func
        except:
            self.flash_attn_func = None

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
            self.qkv_func,
            self.mlp_func,
            self.attention.norm_factor,
            self.flash_attn_func)
