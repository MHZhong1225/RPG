# #    Copyright 2023 Haotian Liu
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.


# import os
# import warnings
# import shutil

# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
# import torch
# from llava.model import *
# from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


# def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
#     kwargs = {"device_map": device_map}

#     if load_8bit:
#         kwargs['load_in_8bit'] = True
#     elif load_4bit:
#         kwargs['load_in_4bit'] = True
#         kwargs['quantization_config'] = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_quant_type='nf4'
#         )
#     else:
#         kwargs['torch_dtype'] = torch.float16

#     if 'llava' in model_name.lower():
#         # Load LLaVA model
#         if 'lora' in model_name.lower() and model_base is None:
#             warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
#         if 'lora' in model_name.lower() and model_base is not None:
#             lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             print('Loading LLaVA from base model...')
#             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
#             token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
#             if model.lm_head.weight.shape[0] != token_num:
#                 model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
#                 model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

#             print('Loading additional LLaVA weights...')
#             if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
#                 non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
#             else:
#                 # this is probably from HF Hub
#                 from huggingface_hub import hf_hub_download
#                 def load_from_hf(repo_id, filename, subfolder=None):
#                     cache_file = hf_hub_download(
#                         repo_id=repo_id,
#                         filename=filename,
#                         subfolder=subfolder)
#                     return torch.load(cache_file, map_location='cpu')
#                 non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
#             non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
#             if any(k.startswith('model.model.') for k in non_lora_trainables):
#                 non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
#             model.load_state_dict(non_lora_trainables, strict=False)

#             from peft import PeftModel
#             print('Loading LoRA weights...')
#             model = PeftModel.from_pretrained(model, model_path)
#             print('Merging LoRA weights...')
#             model = model.merge_and_unload()
#             print('Model is loaded...')
#         elif model_base is not None:
#             # this may be mm projector only
#             print('Loading LLaVA from base model...')
#             if 'mpt' in model_name.lower():
#                 if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
#                     shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#                 cfg_pretrained = AutoConfig.from_pretrained(model_path)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

#             mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
#             mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
#             model.load_state_dict(mm_projector_weights, strict=False)
#         else:
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
#     else:
#         # Load language model
#         if model_base is not None:
#             # PEFT model
#             from peft import PeftModel
#             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
#             model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
#             print(f"Loading LoRA weights from {model_path}")
#             model = PeftModel.from_pretrained(model, model_path)
#             print(f"Merging weights")
#             model = model.merge_and_unload()
#             print('Convert to FP16...')
#             model.to(torch.float16)
#         else:
#             use_fast = False
#             if 'mpt' in model_name.lower():
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
#             else:
#                 tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#                 model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

#     image_processor = None

#     if 'llava' in model_name.lower():
#         mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
#         mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
#         if mm_use_im_patch_token:
#             tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
#         if mm_use_im_start_end:
#             tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
#         model.resize_token_embeddings(len(tokenizer))

#         vision_tower = model.get_vision_tower()
#         if not vision_tower.is_loaded:
#             vision_tower.load_model()
#         vision_tower.to(device=device, dtype=torch.float16)
#         image_processor = vision_tower.image_processor

#     if hasattr(model.config, "max_sequence_length"):
#         context_len = model.config.max_sequence_length
#     else:
#         context_len = 2048

#     return tokenizer, model, image_processor, context_len


#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil
# Removed 'import glob' as _resolve_hf_cache_path function which used it has been removed.

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *  # This imports LlavaLlamaForCausalLM and LlavaMPTForCausalLM
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# The _resolve_hf_cache_path function has been removed.
# It was designed to resolve Hugging Face cache paths, but it caused HFValidationError
# because transformers.from_pretrained expects a repo_id (e.g., 'org/repo_name')
# or a direct local path to a model directory, not the internal cache structure.
# The transformers library handles cache resolution internally when given a valid repo_id.

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    
    # Removed the calls to _resolve_hf_cache_path.
    # The 'model_path' and 'model_base' should now directly be the Hugging Face Hub IDs
    # (e.g., "liuhaotian/llava-v1.5-7b") or direct paths to local model repositories,
    # allowing the transformers library to handle them correctly.

    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    # Critical modification: local_files_only set to False to allow downloading if not fully cached.
    # trust_remote_code is essential for custom LLaVA models.
    common_hf_params = {
        "trust_remote_code": True,  # LLaVA uses custom models, this must be True
        "local_files_only": False   # Set to False to allow downloading from Hugging Face Hub if files are not completely in local cache.
                                    # Change to True only if absolutely sure all model files are present locally and no internet access is desired/permitted.
    }
    
    # Merge kwargs and common_hf_params. Note: if a key exists in both, kwargs will be prioritized.
    # This ensures 'trust_remote_code' and 'local_files_only' are passed correctly.
    kwargs = {**common_hf_params, **kwargs} # Ensure common_hf_params are applied first, then kwargs can override if needed, but in this specific setup, it's just merging.

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            # LLaVA LoRA models
            print(f"DEBUG BUILDER: LLaVA LoRA - AutoConfig.from_pretrained called with model_path: '{model_path}'")
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path, **common_hf_params)
            
            print(f"DEBUG BUILDER: LLaVA LoRA - AutoTokenizer.from_pretrained called with model_base: '{model_base}'")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **common_hf_params)
            print('Loading LLaVA from base model...')
            
            print(f"DEBUG BUILDER: LLaVA LoRA - LlavaLlamaForCausalLM.from_pretrained called with model_base: '{model_base}'")
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        
        elif model_base is not None:
            # This path is for loading LLaVA with a separate vision projector on a base LLM
            print('Loading LLaVA from base model (with separate projector)...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    # Ensure target path exists for config file
                    os.makedirs(model_path, exist_ok=True)
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                
                print(f"DEBUG BUILDER: LLaVA MPT (projector) - AutoTokenizer.from_pretrained called with model_base: '{model_base}'")
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, **common_hf_params)
                print(f"DEBUG BUILDER: LLaVA MPT (projector) - AutoConfig.from_pretrained called with model_path: '{model_path}'")
                cfg_pretrained = AutoConfig.from_pretrained(model_path, **kwargs) # kwargs already includes trust_remote_code etc.
                print(f"DEBUG BUILDER: LLaVA MPT (projector) - LlavaMPTForCausalLM.from_pretrained called with model_base: '{model_base}'")
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                print(f"DEBUG BUILDER: LLaVA Llama (projector) - AutoTokenizer.from_pretrained called with model_base: '{model_base}'")
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **common_hf_params)
                print(f"DEBUG BUILDER: LLaVA Llama (projector) - AutoConfig.from_pretrained called with model_path: '{model_path}'")
                cfg_pretrained = AutoConfig.from_pretrained(model_path, **common_hf_params) # model_path as config source
                print(f"DEBUG BUILDER: LLaVA Llama (projector) - LlavaLlamaForCausalLM.from_pretrained called with model_base: '{model_base}'")
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            # Note: mm_projector_weights are loaded from model_path, which is a file itself, not via from_pretrained
            # This part assumes 'model_path' leads to a local directory containing 'mm_projector.bin'
            print(f"Loading mm_projector.bin from '{model_path}'")
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # Fallback path if no explicit base model is provided for LLaVA (e.g., a fully merged LLaVA model)
            if 'mpt' in model_name.lower():
                print(f"DEBUG BUILDER: LLaVA MPT (fully merged) - AutoTokenizer.from_pretrained called with model_path: '{model_path}'")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **common_hf_params)
                print(f"DEBUG BUILDER: LLaVA MPT (fully merged) - LlavaMPTForCausalLM.from_pretrained called with model_path: '{model_path}'")
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                print(f"DEBUG BUILDER: LLaVA Llama (fully merged) - AutoTokenizer.from_pretrained called with model_path: '{model_path}'")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **common_hf_params)
                print(f"DEBUG BUILDER: LLaVA Llama (fully merged) - LlavaLlamaForCausalLM.from_pretrained called with model_path: '{model_path}'")
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    else:
        # Load language model (non-LLaVA specific)
        if model_base is not None:
            # PEFT model (e.g., LoRA on a non-LLaVA LLM)
            from peft import PeftModel
            print(f"DEBUG BUILDER: Non-LLaVA PEFT - AutoTokenizer.from_pretrained called with model_base: '{model_base}'")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, **common_hf_params)
            print(f"DEBUG BUILDER: Non-LLaVA PEFT - AutoModelForCausalLM.from_pretrained called with model_base: '{model_base}'")
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", **common_hf_params)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            # Pure Language Model (no base, no PEFT, no LLaVA)
            if 'mpt' in model_name.lower():
                print(f"DEBUG BUILDER: Non-LLaVA MPT (base LLM) - AutoTokenizer.from_pretrained called with model_path: '{model_path}'")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **common_hf_params)
                print(f"DEBUG BUILDER: Non-LLaVA MPT (base LLM) - AutoModelForCausalLM.from_pretrained called with model_path: '{model_path}'")
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            else:
                print(f"DEBUG BUILDER: Non-LLaVA Llama (base LLM) - AutoTokenizer.from_pretrained called with model_path: '{model_path}'")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **common_hf_params)
                print(f"DEBUG BUILDER: Non-LLaVA Llama (base LLM) - AutoModelForCausalLM.from_pretrained called with model_path: '{model_path}'")
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len