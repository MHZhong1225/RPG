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
        self.ln = nn.LayerNorm(d_hidden)
        
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
        h_cross = self.ln(h_cross)
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

        processor = LlavaNextProcessor.from_pretrained(lm_name,use_fast=True)
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
            # task_type="CAUSAL_LM",
        )

        residual_lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            # task_type="CAUSAL_LM",
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


@torch.no_grad()
def generate_with_prefix_causal(
    model,
    prefix_vec: torch.Tensor,
    prefix_cf_vec: torch.Tensor,
    adapter_name: str,
    max_new_tokens: int = 128,
    gamma: float = 1.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    反事实推理：
      - factual:     prefix_vec
      - counterfact: prefix_cf_vec （比如“去视觉”的前缀）
    两条序列共享同样的 prompt & 已生成 token，只在 prefix 上做反事实。
    """
    lm = model.lm
    tokenizer = model.tokenizer
    device = prefix_vec.device

    lm.set_adapter(adapter_name)

    # ---------- 1. prefix embeddings ----------
    prefix_embeds_f, prefix_mask_f = model._build_prefix(prefix_vec)    # [1,P,H], [1,P]
    prefix_embeds_cf, prefix_mask_cf = model._build_prefix(prefix_cf_vec)  # [1,P,H], [1,P]

    # ---------- 2. prompt embeddings ----------
    if adapter_name == "template":
        prompt_text = (
            "Write a concise, structured radiology IMPRESSION for a chest X-ray. "
            "Do not list findings. Summarize clinical significance only.\n"
            "Impression: "
        )
    else:
        prompt_text = (
            "Write detailed radiology FINDINGS for a chest X-ray. "
            "Focus on abnormalities. Do not write the impression.\n"
            "Findings: "
        )

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(device)

    input_emb_layer = lm.get_input_embeddings()
    prompt_embeds = input_emb_layer(prompt_ids)  # [1, L, H]
    prompt_mask = torch.ones((1, prompt_embeds.size(1)), dtype=torch.long, device=device)

    # ---------- 3. BOS token ----------
    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_embeds = input_emb_layer(bos_ids)  # [1,1,H]
    bos_mask = torch.ones((1, 1), dtype=torch.long, device=device)

    # ---------- 4. 拼接 prefix + prompt + BOS ----------
    inputs_embeds_f = torch.cat([prefix_embeds_f, prompt_embeds, bos_embeds], dim=1)
    attention_mask_f = torch.cat([prefix_mask_f, prompt_mask, bos_mask], dim=1)

    inputs_embeds_cf = torch.cat([prefix_embeds_cf, prompt_embeds, bos_embeds], dim=1)
    attention_mask_cf = torch.cat([prefix_mask_cf, prompt_mask, bos_mask], dim=1)

    # ---------- 5. 自回归生成（每步跑 factual + counterfactual） ----------
    generated = []
    for _ in range(max_new_tokens):
        # factual 前向
        out_f = lm(
            inputs_embeds=inputs_embeds_f,
            attention_mask=attention_mask_f,
            use_cache=False,
        )
        logits_f = out_f.logits[:, -1, :]  # [1,V]

        # counterfactual 前向
        out_cf = lm(
            inputs_embeds=inputs_embeds_cf,
            attention_mask=attention_mask_cf,
            use_cache=False,
        )
        logits_cf = out_cf.logits[:, -1, :]  # [1,V]

        # 因果融合获取最终 prob
        probs = causal_combine_logits(
            logits_f=logits_f,
            logits_cf=logits_cf,
            gamma=gamma,
            temperature=temperature,
            top_p=top_p,
            eps=1e-5,
        )

        # 采样
        next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
        tok = next_token.item()
        if tok == tokenizer.eos_token_id:
            break

        generated.append(tok)

        # 两条路径都 append 同样的 token（共享语言上下文）
        tok_embeds = input_emb_layer(next_token)  # [1,1,H]

        inputs_embeds_f = torch.cat([inputs_embeds_f, tok_embeds], dim=1)
        attention_mask_f = torch.cat(
            [attention_mask_f, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1,
        )

        inputs_embeds_cf = torch.cat([inputs_embeds_cf, tok_embeds], dim=1)
        attention_mask_cf = torch.cat(
            [attention_mask_cf, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1,
        )

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


def causal_combine_logits(
    logits_f: torch.Tensor,
    logits_cf: torch.Tensor,
    gamma: float = 1.0,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eps: float = 1e-5,
):
    """
    CAUSALMM 风格的后置解码：
      logits_f: factual logits        [1, V]
      logits_cf: counterfactual logits[1, V]

    返回：因果修正后的概率分布 probs [1, V]
    """
    import math
    # 转 float32 稳定一点
    logits_f = logits_f.float()
    logits_cf = logits_cf.float()
    # 论文里的核心：用 Δ = logits_f - logits_cf 做因果校正
    delta = logits_f - logits_cf
    logits_adj = logits_f + gamma * delta - math.log(eps)

    # 为了数值稳定，减去 max
    logits_adj = logits_adj - logits_adj.max(dim=-1, keepdim=True).values

    # temperature
    if temperature > 0:
        logits_adj = logits_adj / temperature

    # top-p (nucleus) 截断，避免尾部奇怪 token
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits_adj, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # cutoff mask
        cutoff_mask = cumulative_probs > top_p
        # 保留第一个超过 top_p 的
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False

        # 把被截掉的 logits 设成 -inf
        sorted_logits = sorted_logits.masked_fill(cutoff_mask, float("-inf"))
        # scatter 回原位置
        logits_adj = torch.full_like(logits_adj, float("-inf"))
        logits_adj.scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits_adj, dim=-1)
    return probs


import torch
import torch.nn.functional as F

@torch.no_grad()
def generate_with_prefix_rpg(
    model,
    prefix_vec_f: torch.Tensor,
    prefix_vec_cf_v: torch.Tensor,
    prefix_vec_cf_l: torch.Tensor,
    adapter_name: str = "residual",
    max_new_tokens: int = 256,
    gamma: float = 0.5,
    epsilon: float = 0.1,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    最终 RPG 推理（对齐论文 CAUSALMM 的 contrastive decoding）：
      logits_f      : factual（真实图像 + 模板先验）
      logits_cf_v   : visual counterfactual（去/扰动视觉）
      logits_cf_l   : language counterfactual（去模板语言先验）
    diffs = (1 + 2γ) * f  - γ * cf_v - γ * cf_l
    cutoff = log(ε) + max(f)
    cf_logits = diffs(mask f < cutoff -> -inf)
    然后对 cf_logits 做 temperature / top_p nucleus sampling。
    """
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TopPLogitsWarper,
        TemperatureLogitsWarper,
    )

    lm = model.lm
    tokenizer = model.tokenizer
    device = prefix_vec_f.device
    input_emb_layer = lm.get_input_embeddings()

    lm.set_adapter(adapter_name)

    # ---------- 1) prefix embeds ----------
    prefix_embeds_f, prefix_mask_f = model._build_prefix(prefix_vec_f)      # [1,P,H], [1,P]
    prefix_embeds_cf_v, prefix_mask_cf_v = model._build_prefix(prefix_vec_cf_v)
    prefix_embeds_cf_l, prefix_mask_cf_l = model._build_prefix(prefix_vec_cf_l)

    # ---------- 2) prompt ----------
    if adapter_name == "template":
        prompt_text = (
            "Write a concise, structured radiology IMPRESSION for a chest X-ray. "
            "Do not list findings. Summarize clinical significance only.\n"
            "Impression: "
        )
    else:
        prompt_text = (
            "Write detailed radiology FINDINGS for a chest X-ray. "
            "Focus on abnormalities. Do not write the impression.\n"
            "Findings: "
        )

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    prompt_embeds = input_emb_layer(prompt_ids)          # [1, L, H]
    prompt_mask = torch.ones((1, prompt_ids.size(1)), dtype=torch.long, device=device)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id  # 给 mistral/llava 做兜底
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_embeds = input_emb_layer(bos_ids)                # [1,1,H]
    bos_mask = torch.ones((1, 1), dtype=torch.long, device=device)

    # Prefix + BOS + Prompt
    inputs_embeds_f = torch.cat([prefix_embeds_f, bos_embeds, prompt_embeds], dim=1)
    attn_f = torch.cat([prefix_mask_f, bos_mask, prompt_mask], dim=1)

    inputs_embeds_cf_v = torch.cat([prefix_embeds_cf_v, bos_embeds, prompt_embeds], dim=1)
    attn_cf_v = torch.cat([prefix_mask_cf_v, bos_mask, prompt_mask], dim=1)

    inputs_embeds_cf_l = torch.cat([prefix_embeds_cf_l, bos_embeds, prompt_embeds], dim=1)
    attn_cf_l = torch.cat([prefix_mask_cf_l, bos_mask, prompt_mask], dim=1)

    # 给 warper 一个“当前已生成 token ids”的占位（TopP/Temp 不依赖具体 ids 内容）
    input_ids_shared = torch.cat([bos_ids, prompt_ids], dim=1)  # [1, 1+L]

    # ---------- 3) 初始化 forward（建立各自 kv cache） ----------
    out_f = lm(inputs_embeds=inputs_embeds_f, attention_mask=attn_f, use_cache=True, return_dict=True)
    pkv_f = out_f.past_key_values
    logits_f = out_f.logits[:, -1, :]  # [1, V]

    out_cf_v = lm(inputs_embeds=inputs_embeds_cf_v, attention_mask=attn_cf_v, use_cache=True, return_dict=True)
    pkv_cf_v = out_cf_v.past_key_values
    logits_cf_v = out_cf_v.logits[:, -1, :]

    out_cf_l = lm(inputs_embeds=inputs_embeds_cf_l, attention_mask=attn_cf_l, use_cache=True, return_dict=True)
    pkv_cf_l = out_cf_l.past_key_values
    logits_cf_l = out_cf_l.logits[:, -1, :]

    # ---------- 4) logits warpers（temperature / top_p） ----------
    warpers = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))

    generated = []

    # ---------- 5) 自回归生成 ----------
    for _ in range(max_new_tokens):
        # float32 稳定
        lf = logits_f.float()
        lcv = logits_cf_v.float()
        lcl = logits_cf_l.float()

        # cutoff = log(eps) + max(f)
        cutoff = torch.log(torch.tensor(epsilon, device=device)) + lf.max(dim=-1, keepdim=True).values

        diffs = (1.0 + 2.0 * gamma) * lf - gamma * lcv - gamma * lcl
        cf_logits = diffs.masked_fill(lf < cutoff, float("-inf"))

        # 应用 warpers（等价论文 logits_warper）
        if len(warpers) > 0:
            cf_logits = warpers(input_ids_shared, cf_logits)

        probs = F.softmax(cf_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)     # [1,1]
        tok = next_token.item()

        if tok == tokenizer.eos_token_id:
            break
        generated.append(tok)

        # 更新 input_ids_shared
        input_ids_shared = torch.cat([input_ids_shared, next_token], dim=1)

        # 三路都 append 同样 token（共享语言上下文）
        tok_embeds = input_emb_layer(next_token)  # [1,1,H]
        one_mask = torch.ones((1, 1), dtype=torch.long, device=device)

        attn_f = torch.cat([attn_f, one_mask], dim=1)
        attn_cf_v = torch.cat([attn_cf_v, one_mask], dim=1)
        attn_cf_l = torch.cat([attn_cf_l, one_mask], dim=1)

        out_f = lm(
            inputs_embeds=tok_embeds,
            attention_mask=attn_f,
            past_key_values=pkv_f,
            use_cache=True,
            return_dict=True,
        )
        pkv_f = out_f.past_key_values
        logits_f = out_f.logits[:, -1, :]

        out_cf_v = lm(
            inputs_embeds=tok_embeds,
            attention_mask=attn_cf_v,
            past_key_values=pkv_cf_v,
            use_cache=True,
            return_dict=True,
        )
        pkv_cf_v = out_cf_v.past_key_values
        logits_cf_v = out_cf_v.logits[:, -1, :]

        out_cf_l = lm(
            inputs_embeds=tok_embeds,
            attention_mask=attn_cf_l,
            past_key_values=pkv_cf_l,
            use_cache=True,
            return_dict=True,
        )
        pkv_cf_l = out_cf_l.past_key_values
        logits_cf_l = out_cf_l.logits[:, -1, :]

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text