# infer.py

import argparse
import os
import json
from PIL import Image
import torch
import torch.nn.functional as F

from model_rpg import RPGModel
from rpg_dataset import RPGDataset


# ==========================================================
# Autoregressive generation with prefix embeddings
# ==========================================================

@torch.no_grad()
def generate_with_prefix(
    model,
    prefix_vec,
    adapter_name: str,
    max_new_tokens: int = 128,
    temperature: float = 0.5,
    top_p: float = 0.9,
):
    """
    prefix_vec: [1, hidden]
    在给定 prefix embedding 的前提下，自回归生成文本。
    """
    lm = model.lm
    tokenizer = model.tokenizer
    device = prefix_vec.device

    lm.set_adapter(adapter_name)

    # ---------- 1. prefix embeddings ----------
    # model._build_prefix 会把 [1, hidden] 变成 [1, P, H] 并返回 mask
    prefix_embeds, prefix_mask = model._build_prefix(prefix_vec)  # [1, P, H], [1, P]

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
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, bos_embeds], dim=1)
    attention_mask = torch.cat([prefix_mask, prompt_mask, bos_mask], dim=1)

    # ---------- 5. 自回归生成 ----------
    generated = []
    for _ in range(max_new_tokens):
        outputs = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, -1, :]  # [1, vocab]

        # 采样策略
        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = (logits / temperature).softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tok = next_token.item()
        if tok == tokenizer.eos_token_id:
            break

        generated.append(tok)

        tok_embeds = input_emb_layer(next_token)  # [1,1,H]
        inputs_embeds = torch.cat([inputs_embeds, tok_embeds], dim=1)

        tok_mask = torch.ones((1, 1), dtype=torch.long, device=device)
        attention_mask = torch.cat([attention_mask, tok_mask], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ==========================================================
#  Template generation (LoRA_t)
# ==========================================================

@torch.no_grad()
def infer_template(model, template_list, max_new_tokens=64):
    """
    template_list: List[str]，从 templates.json 读取的模板句列表。
    对所有模板句做 embedding 平均，得到全局模板先验 P_l。
    """
    device = next(model.parameters()).device

    embs = model.template_encoder(template_list)  # [N, d_t] (在 CPU 上)
    embs = embs.to(device=device, dtype=model.lm.dtype)

    h_prior = embs.mean(dim=0, keepdim=True)  # [1, d_t]

    # 3) 通过 template_proj 映射到 LLM hidden space，作为 prefix
    prefix_vec = model.template_proj(h_prior)  # [1, hidden]

    # 4) 用 template adapter 生成 Impression
    text = generate_with_prefix(
        model=model,
        prefix_vec=prefix_vec,
        adapter_name="template",
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    return text


# ==========================================================
#  Residual / Pathology generation (LoRA_p)
# ==========================================================
@torch.no_grad()
def infer_residual(model, image, template_list, max_new_tokens=128):
    """
    image: 单张 PIL.Image
    template_list: 与 infer_template 相同的模板库，用于得到模板先验并参与 cross-attn。
    """
    device = next(model.parameters()).device

    # 1) 模板先验 embedding（与模板通道共享）
    embs = model.template_encoder(template_list)  # [N, d_t]
    embs = embs.to(device=device, dtype=model.lm.dtype)
    h_prior = embs.mean(dim=0, keepdim=True)     # [1, d_t]

    # 2) 视觉编码：注意这里 image 是 PIL.Image
    h_v = model.visual_encoder(image)          # [1, N_patches, d_v]
    h_v = h_v.to(device=device, dtype=model.lm.dtype)

    # 3) Cross-Attention 对齐，得到跨模态 prefix
    h_cross = model.cross_aligner(h_v, h_prior).squeeze(1)  # [1, hidden]

    # 4) 用 residual adapter 生成 Findings（异常 / 残差）
    text = generate_with_prefix(
        model=model,
        prefix_vec=h_cross,
        adapter_name="residual", 
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    return text


# ==========================================================
# Main
# ==========================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 0) 
    if args.template_file is None:
        args.template_file = f"./rpg_outputs/{args.dataset}/templates.json"
    if args.data_path is None:
        args.data_path = f"./rpg_outputs/{args.dataset}/decomposed_reports.jsonl"

    print(">>> Loading RPG model for inference...")
    model = RPGModel(
        lm_name=args.lm_name,
        clip_model_name=args.clip_name,
        template_model_name=args.template_enc_name,
        lora_ckpt=args.lora_path,
        aux_ckpt=args.aux_path,
    )
    model.to(device)
    model.eval()

    # 1) 加载 templates.json
    print(f">>> Loading template prior from: {args.template_file}")
    with open(args.template_file, "r", encoding="utf-8") as f:
        template_list = json.load(f)   # List[str]

    # 2) 加载数据集（decomposed_reports.jsonl）
    print(f">>> Loading dataset from: {args.data_path}")
    dataset = RPGDataset(
        jsonl_path=args.data_path,
        split="test",
        image_root=f"../datasets/{args.dataset}/images"
    )
    if len(dataset) == 0:
        raise RuntimeError(f"[ERROR] No samples found for split='test' in {args.data_path}")

    idx = args.idx
    if not (0 <= idx < len(dataset)):
        raise IndexError(f"idx={idx} out of range, dataset size={len(dataset)}")

    sample = dataset[idx]
    # RPGDataset.__getitem__ 返回:
    # {
    #   "images": [PIL.Image],
    #   "template": "...",
    #   "pathology": "...",
    #   "orig_report": "..."
    # }
    image = sample["images"][0]
    gt_report = sample["orig_report"]

    print(f">>> Using sample idx = {idx}")

    # 3) 模板通道输出（Impression）
    print("\n>>> Stage 1: Template Generation (LoRA_t)...")
    gen_template = infer_template(
        model,
        template_list=template_list,
        max_new_tokens=args.max_tokens_template,
    )
    print("=== Template Channel Output (IMPRESSION) ===")
    print(gen_template)

    # 4) 残差通道输出（Findings）
    print("\n>>> Stage 2: Residual Path (LoRA_p)...")
    gen_residual = infer_residual(
        model,
        image=image,
        template_list=template_list,
        max_new_tokens=args.max_tokens_residual,
    )
    print("=== Residual Channel Output (FINDINGS) ===")
    print(gen_residual)

    final_report = (
        "FINAL REPORT\n"
        "-------------\n"
        "IMPRESSION (Template Channel):\n"
        f"{gen_template}\n\n"
        "FINDINGS (Residual Channel):\n"
        f"{gen_residual}\n"
    )

    print("\n===== Final RPG Report =====")
    print(final_report)

    print("\n===== GT Report (orig_report) =====")
    print(gt_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="iu_xray")  # iu_xray, MIMIC-CXR
    parser.add_argument("--template_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--lm_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--template_enc_name", type=str, default="pritamdeka/S-BioBert-snli-multinli-stsb")

    parser.add_argument("--lora_path", type=str, default="rpg_llava_lora")
    parser.add_argument("--aux_path", type=str, default="rpg_llava_lora/rpg_aux.pt")

    parser.add_argument("--max_tokens_template", type=int, default=64)
    parser.add_argument("--max_tokens_residual", type=int, default=128)
    parser.add_argument("--idx", type=int, default=0, help="index of test sample")

    args = parser.parse_args()
    main(args)