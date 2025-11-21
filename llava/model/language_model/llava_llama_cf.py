# llava_llama_cf.py

import sys
sys.path.append(".") # Adds higher directory to python modules path.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch_cf import LlavaMetaModel, LlavaMetaForCausalLM
# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import json


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        cf_inputs_embeds=None,     # visual CF (already used)
        cf_inputs_embeds_l=None,   # language CF (new)
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        epsilon=None,
        gamma=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, cf_inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # 构造语言反事实 embedding（轻度扰动）
        if cf_inputs_embeds_l is None and inputs_embeds is not None:
            cf_inputs_embeds_l = self._build_cf_language_embeds(inputs_embeds)

        # 正常
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # 视觉 CF
        cf_outputs_v = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=cf_inputs_embeds,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        # 语言 CF
        cf_outputs_l = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=cf_inputs_embeds_l,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        logits = self.lm_head(outputs[0])
        logits_cf_v = self.lm_head(cf_outputs_v[0])
        logits_cf_l = self.lm_head(cf_outputs_l[0])

        if not return_dict:
            return (logits, logits_cf_v, logits_cf_l)

        return (
            CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values),
            CausalLMOutputWithPast(logits=logits_cf_v, past_key_values=cf_outputs_v.past_key_values),
            CausalLMOutputWithPast(logits=logits_cf_l, past_key_values=cf_outputs_l.past_key_values),
        )

    def _build_cf_language_embeds(self, inputs_embeds, shuffle_ratio: float = 0.3):
        """轻度打乱文本 token 的 embedding 实现语言反事实 (CF-L)"""
        cf = inputs_embeds.clone()
        B, T, H = cf.size()
        for b in range(B):
            # 选取部分 token 打乱
            num = int(T * shuffle_ratio)
            if num > 2:
                idx = torch.randperm(T, device=cf.device)[:num]
                cf[b, idx] = cf[b, idx[torch.randperm(num)]]
        return cf

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs

try:
    AutoConfig.register("llava", LlavaConfig)
except Exception:
    pass
try:
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except Exception:
    pass
