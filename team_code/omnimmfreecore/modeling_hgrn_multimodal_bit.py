# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .encoder import SiglipVisionTower

from .hgrnbitattention import HGRNBitAttention
from .config import HGRNBitMultimodalConfig
from .utils import RecurrentCache
from .rmsnorm import RMSNorm, RMSNormLinear
from .fusedcrossentropy import FusedCrossEntropyLoss
from .activation import swiglu
from .fusedbitnet import FusedBitLinear as BitLinear

logger = logging.get_logger(__name__)


class HGRNBitMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
    ) -> HGRNBitMLP:
        super().__init__()

        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = BitLinear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = BitLinear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, y = y.chunk(2, -1)
        z = self.down_proj(swiglu(gate, y))
        return z


class HGRNBitBlock(nn.Module):
    def __init__(self, config: HGRNBitMultimodalConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn = HGRNBitAttention(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            expand_ratio=config.expand_ratio,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            share_conv_kernel=config.share_conv_kernel,
            layernorm_eps=config.rms_norm_eps,
            layer_idx=layer_idx,
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = HGRNBitMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        lower_bound: Optional[torch.Tensor] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            lower_bound=lower_bound,
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class HGRNBitMultimodalPreTrainedModel(PreTrainedModel):

    config_class = HGRNBitMultimodalConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["HGRNBitBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(
                            num_residuals_per_layer * self.config.num_hidden_layers
                        )


class HGRNBitMultimodalModel(HGRNBitMultimodalPreTrainedModel):

    def __init__(self, config: HGRNBitMultimodalConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.video_feat_size = config.video_frame_size * config.video_channels
        self.audio_feat_size = config.audio_feat_size
        
        self.vision_tower = SiglipVisionTower("google/siglip-base-patch16-224", mm_vision_select_layer=-2)

        self.text_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        self.video_proj = nn.Linear(self.video_feat_size, config.hidden_size)
        # self.audio_proj = nn.Linear(self.audio_feat_size, config.hidden_size)

        if config.use_lower_bound:
            self.lower_bounds = nn.Parameter(
                torch.zeros(config.num_hidden_layers, config.hidden_size)
            )

        self.layers = nn.ModuleList(
            [
                HGRNBitBlock(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.post_init()
        
    # TODO
    def get_vision_tower(self):
        pass
    
    def get_input_embeddings(self):
        return self.text_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        video_ids: Optional[torch.Tensor] = None,
        audio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is None and video_ids is None and audio_ids is None:
            raise ValueError(
                "At least one of the inputs (text, video, audio) must be provided."
            )

        text_embeds = self.text_embeddings(input_ids) if input_ids is not None else None
        video_embeds = self.video_proj(input_ids) if video_ids is not None else None
        # audio_embeds = self.audio_proj(input_ids) if audio_ids is not None else None

        if self.config.fusion_method == "concat":
            hidden_states = torch.cat(
                [
                    emb
                    for emb in [text_embeds, video_embeds] #, audio_embeds]
                    if emb is not None
                ],
                dim=1,
            )
        else:
            raise ValueError(
                f"Fusion method {self.config.fusion_method} not supported."
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        if self.config.use_lower_bound:
            lower_bounds = self.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]

        for i, layer in enumerate(self.layers):
            lower_bound = lower_bounds[i] if self.config.use_lower_bound else None

            hidden_states, attentions, past_key_values = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                lower_bound=lower_bound,
            )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                x
                for x in [hidden_states, past_key_values, all_hidden_states, all_attns]
                if x is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class HGRNBitForCausalLM(HGRNBitMultimodalPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HGRNBitMultimodalModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if "past_key_values" in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(
                    past_key_values, input_ids.shape[1] - 1
                )
            input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = (
                FusedCrossEntropyLoss(inplace_backward=True)
                if self.config.fuse_cross_entropy
                else nn.CrossEntropyLoss()
            )

            labels = labels.to(logits.device)
            labels = torch.cat(
                (
                    labels[..., 1:],
                    torch.full_like(labels[:, :1], loss_fct.ignore_index),
                ),
                1,
            )

            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
