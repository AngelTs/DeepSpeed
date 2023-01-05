'''
Copyright 2022 The Microsoft DeepSpeed Team
'''

import math
import torch
from torch.autograd import Function
from ... import op_builder
import torch.nn as nn
from deepspeed import comm as dist

minus_inf = -10000.0
inference_cuda_module = None


class DeepSpeedSelfAttentionFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        input_mask,
        head_mask,
        layer_past,
        get_present,
        encoder_hidden_states,
        encoder_attention_mask,
        output_attentions,
        norm_w,
        norm_b,
        config,
        attn_qkvw,
        attn_qkvb,
        num_attention_heads_per_partition,
        norm_factor,
        hidden_size_per_partition,
        attn_ow,
        attn_ob,
        mp_group,
        q_scales,
        q_groups,
        merge_count,
        qkv_merging,
        score_context_func,
        alibi,
        qkv_func,
        vector_matmul_func,
        linear_func,
    ):
        def _transpose_for_scores(x, key=False, reshape=False):
            attention_head_size = x.shape[-1] // num_attention_heads_per_partition
            new_x_shape = x.size()[:-1] + (
                num_attention_heads_per_partition,
                attention_head_size,
            )
            x_1 = x.view(*new_x_shape)
            if key:
                x_1 = x_1.permute(0, 2, 3, 1)
            else:
                x_1 = x_1.permute(0, 2, 1, 3)
            if reshape:
                return x_1.reshape(x.shape)
            return x_1.contiguous()

        def _transpose_for_context(x):
            x = x.permute(0, 2, 1, 3).contiguous()
            new_x_layer_shape = x.size()[:-2] + (hidden_size_per_partition,)
            return x.view(*new_x_layer_shape).contiguous()

        def compute_attention(qkv_out, input_mask):
            no_masking = input_mask is None
            if no_masking:
                input_mask = torch.empty(1)
            alibi_offset = (
                dist.get_rank() * num_attention_heads_per_partition
                if dist.is_initialized()
                else 0
            )

                if config.rotary_dim > 0:
                    mixed_query, key_layer = inference_cuda_module.apply_rotary_pos_emb(
                        mixed_query,
                        key_layer,
                        config.rotary_dim,
                        0 if layer_past is None else layer_past[0].shape[-2],
                        num_attention_heads_per_partition,
                        config.rotate_half,
                        config.rotate_every_two)
                if layer_past is not None:
                    past_key, past_value = layer_past
                    key_layer = torch.cat((past_key.type_as(key_layer),
                                           key_layer),
                                          dim=-2)
                    value_layer = torch.cat((past_value.type_as(value_layer),
                                             value_layer),
                                            dim=-2)
                presents = (key_layer, value_layer)
                mixed_query = _transpose_for_scores(mixed_query, False, True)
                key_layer = _transpose_for_scores(
                    key_layer,
                    True,
                    True) / (norm_factor if config.scale_attention else 1.0)
                value_layer = _transpose_for_scores(value_layer, False, True)
                if layer_past is None:
                    attn_key_value = score_context_func(
                        mixed_query,
                        key_layer,
                        torch.empty(1),
                        ((1 - input_mask).half() *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        value_layer,
                        torch.empty(1),
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        (not unfused_mode),  # noqa: F821
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking)
                else:
                    attn_key_value = score_context_func(
                        mixed_query,
                        (key_layer if unfused_mode else past_key.type_as(key_layer)),  # noqa: F821
                        key_layer,
                        ((1 - input_mask).half() *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        (value_layer
                         if unfused_mode else past_value.type_as(value_layer)),  # noqa: F821
                        value_layer,
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        (not unfused_mode),  # noqa: F821
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking)
                if unfused_mode:  # noqa: F821
                    context_layer, _, _ = attn_key_value
                else:
                    context_layer, key_layer, value_layer = attn_key_value

                # Transpose Context
                context_layer = _transpose_for_context(context_layer)

                return context_layer, presents[0], presents[1] # atten_output, key_layer, value_layer
            else:
                # Note: This modification is added for the BLOOM-176B model and will be removed later!
                if config.bigscience_bloom:
                    context_layer, presents = backup_attention(qkv_out, layer_past, alibi, input_mask, norm_factor)
                    return context_layer, presents[0], presents[1] #key_layer, value_layer
                else:
                    if alibi is not None:
                        batch_heads = qkv_out.shape[0] * num_attention_heads_per_partition
                        offset = dist.get_rank() * batch_heads if dist.is_initialized(
                        ) else 0
                        sliced_alibi = alibi[offset:batch_heads + offset, :, :]

                    attn_key_value = score_context_func(
                        qkv_out,
                        ((1 - input_mask).to(qkv_out.dype) *
                         minus_inf) if input_mask.dtype == torch.int64 else input_mask,
                        config.rotary_dim,
                        config.rotate_half,
                        config.rotate_every_two,
                        num_attention_heads_per_partition,
                        (1 / norm_factor if config.scale_attention else 1.0),
                        config.triangular_masking,
                        config.local_attention,
                        config.window_size,
                        no_masking,
                        config.layer_id,
                        DeepSpeedSelfAttention.num_layers,
                        sliced_alibi if alibi is not None else torch.empty(1))
                    context_layer, key_layer, value_layer = attn_key_value
                    return context_layer, key_layer, value_layer

        def selfAttention_fp():
            if not config.pre_layer_norm:
                linear_func = inference_cuda_module.linear_layer_fp16 if config.fp16 else \
                                    inference_cuda_module.linear_layer_fp32
                qkv_out = linear_func(input,
                                      attn_qkvw,
                                      attn_qkvb,
                                      attn_qkvb is not None,
                                      False,
                                      num_attention_heads_per_partition)
            else:
                qkv_out = qkv_func(input,
                                   attn_qkvw,
                                   attn_qkvw.scale,
                                   (attn_qkvb if attn_qkvb is not None else norm_b),
                                   norm_w,
                                   norm_b,
                                   config.epsilon,
                                   (attn_qkvb is not None),
                                   DeepSpeedSelfAttention.num_layers,
                                   config.bigscience_bloom,
                                   config.mp_size,
                                   dist.get_rank() if dist.is_initialized() else 0,
                                   config.enable_qkv_quantization)
            context_layer, key_layer, value_layer = compute_attention(qkv_out[0] if isinstance(qkv_out, list) else qkv_out, input_mask)

            output = vector_matmul_func(context_layer,
                                        attn_ow,
                                        False,
                                        attn_ow.scale,
                                        config.q_int8)
            return output, key_layer, value_layer, context_layer, qkv_out[-1]

        output, key_layer, value_layer, context_layer, inp_norm = selfAttention_fp()
        if (
            config.mlp_after_attn
            and mp_group is not None
            and dist.get_world_size(group=mp_group) > 1
        ):
            dist.all_reduce(output, group=mp_group)

        return (output, key_layer, value_layer, context_layer, inp_norm)

    @staticmethod
    def backward(ctx, grad_output, grad_output1, grad_output2, grad_output3):
        raise RuntimeError(
            "You are running with DeepSpeed Inference mode. \
                            Please switch to Training mode for running backward!"
        )


class DeepSpeedSelfAttention(nn.Module):
    num_layers = 0

    def __init__(
        self,
        config,
        mp_group=None,
        q_scales=None,
        q_groups=1,
        merge_count=1,
        qkv_merging=False,
    ):
        super(DeepSpeedSelfAttention, self).__init__()
        self.config = config
        data_type = (
            torch.int8 if config.q_int else torch.half if config.fp16 else torch.float
        )
        data_type_fp = torch.half if config.fp16 else torch.float
        self.config.layer_id = DeepSpeedSelfAttention.num_layers
        DeepSpeedSelfAttention.num_layers = DeepSpeedSelfAttention.num_layers + 1
        device = torch.cuda.current_device()  #if config.bigscience_bloom else 'cpu'
        qkv_size_per_partition = (self.config.hidden_size // self.config.mp_size) * 3
        # half_size = config.q_int and config.q_bits == 4
        half_size = False
        self.attn_qkvw = nn.Parameter(
            torch.empty(
                self.config.hidden_size // 2
                if half_size and config.enable_qkv_quantization
                else self.config.hidden_size,
                qkv_size_per_partition,
                dtype=data_type,
                device=device,
            ),
            requires_grad=False,
        )
        self.attn_qkvb = nn.Parameter(
            torch.empty(qkv_size_per_partition, dtype=data_type_fp, device=device),
            requires_grad=False,
        )
        out_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.attn_ow = nn.Parameter(
            torch.empty(
                (out_size_per_partition // 2) if half_size else out_size_per_partition,
                self.config.hidden_size,
                dtype=data_type,
                device=device,
            ),
            requires_grad=False,
        )
        self.attn_ob = nn.Parameter(
            torch.empty(self.config.hidden_size, dtype=data_type_fp, device=device),
            requires_grad=False,
        )

        self.num_attention_heads_per_partition = (
            self.config.heads // self.config.mp_size
        )
        self.hidden_size_per_partition = self.config.hidden_size // self.config.mp_size
        self.hidden_size_per_attention_head = (
            self.config.hidden_size // self.config.heads
        )

        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        self.mp_group = mp_group

        # used for quantization
        self.q_scales = q_scales
        self.q_groups = q_groups
        self.merge_count = int(math.log2(merge_count))

        self.norm_factor = math.sqrt(
            math.sqrt(self.config.hidden_size // self.config.heads)
        )
        self.qkv_merging = qkv_merging

        if self.config.scale_attn_by_inverse_layer_idx is True:
            self.norm_factor *= math.sqrt(self.config.layer_id + 1)
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/gpt2/modeling_gpt2.py#L191

        global inference_cuda_module
        if inference_cuda_module is None:
            builder = op_builder.InferenceBuilder()
            inference_cuda_module = builder.load()

        self.score_context_func = inference_cuda_module.softmax_context_fp32 if (not config.fp16) else \
                                    inference_cuda_module.softmax_context_fp16

        self.score_context_func = (
            inference_cuda_module.softmax_context_fp32
            if (not config.fp16)
            else inference_cuda_module.softmax_context_fp16
        )

    def forward(
        self,
        input,
        input_mask,
        head_mask=None,
        layer_past=None,
        get_present=False,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        norm_w=None,
        norm_b=None,
        alibi=None,
    ):
        output = DeepSpeedSelfAttentionFunction.apply(
            input,
            input_mask,
            head_mask,
            layer_past,
            get_present,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            norm_w,
            norm_b,
            self.config,
            self.attn_qkvw,
            self.attn_qkvb,
            self.num_attention_heads_per_partition,
            self.norm_factor,
            self.hidden_size_per_partition,
            self.attn_ow,
            self.attn_ob,
            self.mp_group,
            self.q_scales,
            self.q_groups,
            self.merge_count,
            self.qkv_merging,
            self.score_context_func,
            alibi,
            self.qkv_func,
            self.vector_matmul_func,
            self.linear_func,
        )

        return output
