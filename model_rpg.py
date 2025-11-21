# model_rag.py

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image

from transformers import (
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    CLIPModel,
    CLIPProcessor,
)
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, PeftModel


#########################################
# 1. 视觉编码器：CLIP
#########################################

class VisualEncoder(nn.Module):
    """
    支持两种输入：
    1) 推理时：单张 PIL.Image 或 list[PIL.Image]
    2) 训练时：batch_images = list[list[PIL.Image]]
       例如：
           [
               [img1],         # sample 1
               [img1, img2],   # sample 2
           ]
    输出：
        h_v: [B, N_total_patches, D]
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        for p in self.clip.parameters():
            p.requires_grad = False

    def _encode_single_image(self, img, device):
        """Encode a single PIL.Image → [1, N, D]"""
        inputs = self.processor(images=img, return_tensors="pt").to(device)
        outputs = self.clip.vision_model(pixel_values=inputs["pixel_values"])
        return outputs.last_hidden_state  # [1, N, D]

    def forward(self, x):
        """
        x 可以是：
        - 单张 PIL.Image
        - list[PIL.Image]
        - list[list[PIL.Image]]
        """

        device = next(self.clip.parameters()).device

        # ---------- Case 1: 单张图片 ----------
        if isinstance(x, Image.Image):
            return self._encode_single_image(x, device)  # [1, N, D]

        # ---------- Case 2: list[PIL.Image] ----------
        if isinstance(x, list) and isinstance(x[0], Image.Image):
            # 多视角（比如 AP + LAT）
            view_embeds = [self._encode_single_image(img, device) for img in x]
            return torch.cat(view_embeds, dim=1)  # [1, N_total, D]

        # ---------- Case 3: list[list[PIL.Image]] (训练时 batch) ----------
        if isinstance(x, list) and isinstance(x[0], list):
            batch_embeds = []
            for img_list in x:
                view_embeds = [self._encode_single_image(img, device) for img in img_list]
                h_v = torch.cat(view_embeds, dim=1)  # [1, N_total, D]
                batch_embeds.append(h_v)

            return torch.cat(batch_embeds, dim=0)  # [B, N_total, D]

        raise TypeError(f"Unsupported input type for VisualEncoder: {type(x)}")


#########################################
# 2. 模板编码器：BioBERT SBERT
#########################################

class TemplateEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "pritamdeka/S-BioBert-snli-multinli-stsb"
    ):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    def forward(self, template_texts):
        if isinstance(template_texts, str):
            template_texts = [template_texts]
        embs = self.model.encode(
            template_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        return embs  # [B, d_t]

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()


#########################################
# 3. 模板-影像 Cross-Attention
#########################################

class CrossAttentionAligner(nn.Module):
    def __init__(self, d_v: int, d_t: int, d_hidden: int, n_heads: int = 8):
        super().__init__()
        self.proj_v = nn.Linear(d_v, d_hidden)
        self.proj_t = nn.Linear(d_t, d_hidden)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=n_heads,
            batch_first=True
        )

    def forward(self, h_v, h_t):
        # h_v: [B, N_patches, d_v]
        # h_t: [B, d_t]
        B, N, _ = h_v.shape
        h_v_proj = self.proj_v(h_v)              # [B, N, hidden]
        h_t_proj = self.proj_t(h_t).unsqueeze(1) # [B, 1, hidden]

        h_cross, _ = self.attn(
            query=h_t_proj,
            key=h_v_proj,
            value=h_v_proj
        )  # [B, 1, hidden]

        return h_cross


#########################################
# 4. RPG 主模型
#########################################

class RPGModel(nn.Module):
    """
    双 LoRA：
      - adapter "template": 模板通道 LoRA_t
      - adapter "residual": 残差通道 LoRA_p
    """

    def __init__(
        self,
        lm_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        template_model_name: str = "pritamdeka/S-BioBert-snli-multinli-stsb",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules=None,
        # ===== 推理 加载已有 LoRA & 对齐模块 =====
        lora_ckpt: str = None,
        aux_ckpt: str = None,   # 保存 cross_aligner + template_proj 的 ckpt 路径
    ):
        super().__init__()

        # 1) 加载 LLaVA-Next 文本侧
        base_lm = LlavaNextForConditionalGeneration.from_pretrained(
            lm_name,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        processor = LlavaNextProcessor.from_pretrained(lm_name)
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if hasattr(base_lm.config, "text_config"):
            hidden_size = base_lm.config.text_config.hidden_size
        else:
            hidden_size = base_lm.config.hidden_size

        # 2) LoRA 配置
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        template_lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        residual_lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 3) 构建双 LoRA 
        if lora_ckpt is None:
            # 训练模式：从头创建 template + residual
            # 第一个 adapter (命名"template")
            self.lm = get_peft_model(base_lm, template_lora, adapter_name="template") 
            # residual adapter
            self.lm.add_adapter("residual", residual_lora)
        else:
            # 推理模式：从 ckpt 恢复
            # 假设 lora_ckpt 指向包含 template/ 和 residual/ 的父目录
            print(f">>> Loading LoRA adapters from: {lora_ckpt}")
            
            # 1. 加载 Template Adapter (作为基础 PEFT 模型)
            template_path = os.path.join(lora_ckpt, "template")
            self.lm = PeftModel.from_pretrained(
                base_lm,
                template_path,
                adapter_name="template",
                torch_dtype=torch.bfloat16,
            )
            
            # 2. 加载 Residual Adapter
            residual_path = os.path.join(lora_ckpt, "residual")
            self.lm.load_adapter(residual_path, adapter_name="residual")

        # 默认使用 template adapter
        self.lm.set_adapter("template")

        print(">>> 已加载 adapter")
        self.lm.print_trainable_parameters()

        # 4) 视觉 / 模板 / 对齐模块
        self.visual_encoder = VisualEncoder(clip_model_name=clip_model_name)
        self.template_encoder = TemplateEncoder(model_name=template_model_name)

        d_v = self.visual_encoder.clip.vision_model.config.hidden_size
        d_t = self.template_encoder.dim

        self.cross_aligner = CrossAttentionAligner(
            d_v=d_v,
            d_t=d_t,
            d_hidden=hidden_size,
            n_heads=8
        )

        self.template_proj = nn.Linear(d_t, hidden_size, bias=False)

        # 与 LM dtype 对齐
        model_dtype = self.lm.dtype
        self.cross_aligner.to(dtype=model_dtype)
        self.template_proj.to(dtype=model_dtype)

        # 若提供 aux_ckpt（推理时），加载 cross_aligner & template_proj
        if aux_ckpt is not None:
            ckpt = torch.load(aux_ckpt, map_location="cpu")
            self.cross_aligner.load_state_dict(ckpt["cross_aligner"])
            self.template_proj.load_state_dict(ckpt["template_proj"])

    # ---------------------------
    # 构造 prefix embeddings
    # ---------------------------

    def _build_prefix(self, prefix_vec: torch.Tensor):
        if prefix_vec.dim() == 2:
            prefix_vec = prefix_vec.unsqueeze(1)
        prefix_embeds = prefix_vec
        prefix_mask = torch.ones(
            (prefix_embeds.size(0), 1),
            dtype=torch.long,
            device=prefix_embeds.device
        )
        return prefix_embeds, prefix_mask

    # ---------------------------
    # 5. 模板：LoRA_t
    # ---------------------------

    def forward_template(
        self,
        template_texts,
        labels_texts=None,
        max_length: int = 256,
        return_prefix: bool = False,
    ):
        device = next(self.lm.parameters()).device

        if labels_texts is None:
            labels_texts = template_texts

        # 1) 模板编码
        with torch.no_grad():
            h_t = self.template_encoder(template_texts)
        h_t = h_t.to(device=device, dtype=self.lm.dtype)

        batch_size = h_t.size(0)

        # 2) 文本编码
        enc = self.tokenizer(
            labels_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # 3) 用 h_t 构造 prefix
        prefix_vec = self.template_proj(h_t)  # [B, hidden]
        prefix_embeds, prefix_mask = self._build_prefix(prefix_vec)

        # 4) token embeddings
        tok_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        labels = input_ids.clone()
        ignore_prefix = torch.full(
            (batch_size, 1),
            -100,
            dtype=torch.long,
            device=device
        )
        labels = torch.cat([ignore_prefix, labels], dim=1)

        self.lm.set_adapter("template")
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = outputs.loss

        if return_prefix:
            # 加性分解
            return loss, outputs, prefix_vec
        return loss, outputs

    # 6. 残差通道（factual）：LoRA_p ("residual")
    def forward_residual(
        self,
        images,
        template_texts,
        pathology_texts,
        max_length: int = 256,
        use_labels: bool = True,
        return_hidden: bool = True,
        return_prefix: bool = False,
    ):
        device = next(self.lm.parameters()).device

        # 1) 视觉编码
        h_v = self.visual_encoder(images).to(device=device, dtype=self.lm.dtype)

        # 2) 模板编码
        with torch.no_grad():
            h_t = self.template_encoder(template_texts)
        h_t = h_t.to(device=device, dtype=self.lm.dtype)

        batch_size = h_t.size(0)

        # 3) cross-attn: 这里的 h_cross ~ R_p 的前身
        h_cross = self.cross_aligner(h_v, h_t)  # [B, 1, hidden]

        # 4) pathology 文本
        enc = self.tokenizer(
            pathology_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        prefix_vec = h_cross.squeeze(1)          # 这一行可以看作 R_p
        prefix_embeds, prefix_mask = self._build_prefix(prefix_vec)

        tok_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, tok_embeds], dim=1)
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        labels = None
        if use_labels:
            labels = input_ids.clone()
            ignore_prefix = torch.full(
                (batch_size, 1),
                -100,
                dtype=torch.long,
                device=device
            )
            labels = torch.cat([ignore_prefix, labels], dim=1)

        self.lm.set_adapter("residual")
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels,
            output_hidden_states=return_hidden,
            return_dict=True,
        )
        if return_prefix:
            # 算 Δr_p / 因果正则
            return outputs, h_v, h_t, prefix_vec
        return outputs, h_v, h_t

    # ---------------------------
    # 7. 隐式反事实：在 h_v 上做操作
    # ---------------------------

    def make_implicit_counterfactual(
        self,
        h_v,
        mode: str = "patch_drop",
        drop_ratio: float = 0.3,
    ):
        if mode == "patch_drop":
            B, N, D = h_v.shape
            device = h_v.device
            keep_mask = (torch.rand(B, N, 1, device=device) > drop_ratio).to(h_v.dtype)
            h_v_cf = h_v * keep_mask
            return h_v_cf

        elif mode == "patch_shuffle":
            B, N, D = h_v.shape
            device = h_v.device
            h_v_cf = torch.zeros_like(h_v)
            for b in range(B):
                perm = torch.randperm(N, device=device)
                h_v_cf[b] = h_v[b, perm]
            return h_v_cf

        return h_v

    # ---------------------------
    # 8. 残差通道（implicit CF）：LoRA_p
    # ---------------------------

    def forward_residual_implicit_cf(
        self,
        h_v,
        h_t,
        pathology_texts,
        max_length: int = 256,
        use_labels: bool = False,
        return_hidden: bool = True,
        cf_mode: str = "patch_drop",
        drop_ratio: float = 0.3,
        return_prefix: bool = False,
    ):
        device = next(self.lm.parameters()).device

        # 1) 构造 h_v_cf
        h_v_cf = self.make_implicit_counterfactual(
            h_v, mode=cf_mode, drop_ratio=drop_ratio
        )

        # 2) cross-attn with same h_t
        h_t = h_t.to(device=device, dtype=self.lm.dtype)
        h_cross_cf = self.cross_aligner(h_v_cf, h_t)  # [B,1,hidden]

        batch_size = h_t.size(0)

        # 3) pathology 文本
        enc = self.tokenizer(
            pathology_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        prefix_vec_cf = h_cross_cf.squeeze(1)
        prefix_embeds_cf, prefix_mask_cf = self._build_prefix(prefix_vec_cf)

        tok_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds_cf = torch.cat([prefix_embeds_cf, tok_embeds], dim=1)
        extended_mask_cf = torch.cat([prefix_mask_cf, attention_mask], dim=1)

        labels_cf = None
        if use_labels:
            labels_cf = input_ids.clone()
            ignore_prefix_cf = torch.full(
                (batch_size, 1),
                -100,
                dtype=torch.long,
                device=device
            )
            labels_cf = torch.cat([ignore_prefix_cf, labels_cf], dim=1)

        self.lm.set_adapter("residual")
        outputs_cf = self.lm(
            inputs_embeds=inputs_embeds_cf,
            attention_mask=extended_mask_cf,
            labels=labels_cf,
            output_hidden_states=return_hidden,
            return_dict=True,
        )
        if return_prefix:
            return outputs_cf, prefix_vec_cf
        return outputs_cf