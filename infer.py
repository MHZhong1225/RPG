
# infer_rpg_causalmm.py
import argparse
import os
import json
import types
from PIL import Image

import torch
import torch.nn.functional as F
from tqdm import tqdm

from model_rpg import RPGModel
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
)

# ==========================================================
# CausalMM Core: Language Self-Attention Intervention Patch
# ==========================================================

class CausalAttentionContext:
    """
    推理期语言注意力干预（对应论文 do(A_t = A*_t) 的工程近似）。
    只在进入该上下文时 patch LLM 的 self-attn forward，
    退出时恢复原始 forward。
    mode: 'original' | 'random' | 'uniform' | 'shuffle'
    """
    def __init__(self, model, mode="original", scaling_factor=1.0, layer_group="middle"):
        self.model = model.lm
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.layer_group = layer_group

        self.original_forward_funcs = {}
        self.modules_to_patch = []

        # 找到 decoder layers
        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "layers"):
            layers = self.model.language_model.model.layers
        else:
            # fallback: patch all modules named self_attn
            for name, module in self.model.named_modules():
                if "self_attn" in name and hasattr(module, "forward"):
                    self.modules_to_patch.append(module)
            layers = None

        if layers is not None:
            num_layers = len(layers)
            third = max(1, num_layers // 3)

            if layer_group == "all":
                idxs = list(range(num_layers))
            elif layer_group == "early":
                idxs = list(range(0, third))
            elif layer_group == "late":
                idxs = list(range(num_layers - third, num_layers))
            else:  # middle
                start = (num_layers - third) // 2
                idxs = list(range(start, start + third))

            for i in idxs:
                layer = layers[i]
                attn = getattr(layer, "self_attn", None)
                if attn is not None and hasattr(attn, "forward"):
                    self.modules_to_patch.append(attn)

    def __enter__(self):
        if self.mode == "original":
            return self

        for module in self.modules_to_patch:
            self.original_forward_funcs[module] = module.forward
            module.forward = types.MethodType(self._get_intervened_forward(module), module)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == "original":
            return

        for module in self.modules_to_patch:
            if module in self.original_forward_funcs:
                module.forward = self.original_forward_funcs[module]
        self.original_forward_funcs.clear()

    def _get_intervened_forward(self, original_module):
        original_forward = self.original_forward_funcs[original_module]
        mode = self.mode
        scale = self.scaling_factor

        def intervened_forward(self, *args, **kwargs):
            outputs = original_forward(*args, **kwargs)

            if isinstance(outputs, tuple):
                attn_output = outputs[0]
            else:
                attn_output = outputs

            if mode == "random":
                noise = torch.randn_like(attn_output)
                mean = attn_output.mean()
                std = attn_output.std()
                intervened_output = (noise - noise.mean()) / (noise.std() + 1e-6) * std + mean
                intervened_output = intervened_output * scale

            elif mode == "uniform":
                mean_output = attn_output.mean(dim=1, keepdim=True).expand_as(attn_output)
                intervened_output = mean_output * scale

            elif mode == "shuffle":
                B, T, H = attn_output.shape
                perm = torch.randperm(T, device=attn_output.device)
                intervened_output = attn_output[:, perm, :] * scale

            else:
                return outputs

            if isinstance(outputs, tuple):
                return (intervened_output,) + outputs[1:]
            return intervened_output

        return intervened_forward


# ==========================================================
# Final RPG + CausalMM decoding (3-branch)
# ==========================================================

@torch.no_grad()
def generate_rpg_causalmm(
    model,
    prefix_f: torch.Tensor,
    prefix_cf_v: torch.Tensor,
    adapter_name: str = "residual",
    max_new_tokens: int = 256,
    gamma: float = 0.5,
    epsilon: float = 0.1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    lang_cf_type: str = "random",
    lang_cf_layers: str = "middle",
):
    """
    三路 logits:
      factual:    ℓ
      visual CF:  ℓ_cf_v  (implicit visual counterfactual via prefix_cf_v)
      language CF ℓ_cf_l  (do(A_t=A*) via attention patch)
    diffs = (1+2γ)ℓ - γℓ_cf_v - γℓ_cf_l
    cutoff = log(ε)+max(ℓ)
    在 final_logits 上做 temperature/top_p nucleus sampling.
    """
    lm = model.lm
    tokenizer = model.tokenizer
    device = prefix_f.device
    input_emb_layer = lm.get_input_embeddings()

    lm.set_adapter(adapter_name)

    prompt_text = (
        "Write detailed radiology FINDINGS for a chest X-ray. "
        "Focus on abnormalities. Do not write the impression.\n"
        "Findings: "
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_embeds = input_emb_layer(prompt_ids)
    prompt_mask = torch.ones((1, prompt_ids.size(1)), dtype=torch.long, device=device)

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_embeds = input_emb_layer(bos_ids)
    bos_mask = torch.ones((1, 1), dtype=torch.long, device=device)

    p_emb_f, p_mask_f = model._build_prefix(prefix_f)
    p_emb_v, p_mask_v = model._build_prefix(prefix_cf_v)

    inputs_embeds_f = torch.cat([p_emb_f, bos_embeds, prompt_embeds], dim=1)
    attn_f = torch.cat([p_mask_f, bos_mask, prompt_mask], dim=1)

    inputs_embeds_v = torch.cat([p_emb_v, bos_embeds, prompt_embeds], dim=1)
    attn_v = torch.cat([p_mask_v, bos_mask, prompt_mask], dim=1)

    inputs_embeds_l = inputs_embeds_f
    attn_l = attn_f

    warpers = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))

    out_f = lm(inputs_embeds=inputs_embeds_f, attention_mask=attn_f, use_cache=True, return_dict=True)
    pkv_f = out_f.past_key_values
    logits_f = out_f.logits[:, -1, :]

    out_v = lm(inputs_embeds=inputs_embeds_v, attention_mask=attn_v, use_cache=True, return_dict=True)
    pkv_v = out_v.past_key_values
    logits_v = out_v.logits[:, -1, :]

    with CausalAttentionContext(model, mode=lang_cf_type, layer_group=lang_cf_layers):
        out_l = lm(inputs_embeds=inputs_embeds_l, attention_mask=attn_l, use_cache=True, return_dict=True)
    pkv_l = out_l.past_key_values
    logits_l = out_l.logits[:, -1, :]

    input_ids_shared = torch.cat([bos_ids, prompt_ids], dim=1)
    generated = []

    for _ in tqdm(range(max_new_tokens), desc="RPG-CausalMM Decoding"):
        lf = logits_f.float()
        lcv = logits_v.float()
        lcl = logits_l.float()

        cutoff = torch.log(torch.tensor(epsilon, device=device)) + lf.max(dim=-1, keepdim=True).values
        diffs = (1.0 + 2.0 * gamma) * lf - gamma * lcv - gamma * lcl
        final_logits = diffs.masked_fill(lf < cutoff, float("-inf"))

        if len(warpers) > 0:
            final_logits = warpers(input_ids_shared, final_logits)

        probs = torch.softmax(final_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tok = next_token.item()
        if tok == tokenizer.eos_token_id:
            break
        generated.append(tok)

        input_ids_shared = torch.cat([input_ids_shared, next_token], dim=1)
        tok_emb = input_emb_layer(next_token)
        one_mask = torch.ones((1, 1), dtype=torch.long, device=device)

        attn_f = torch.cat([attn_f, one_mask], dim=1)
        attn_v = torch.cat([attn_v, one_mask], dim=1)
        attn_l = torch.cat([attn_l, one_mask], dim=1)

        out_f = lm(inputs_embeds=tok_emb, attention_mask=attn_f, past_key_values=pkv_f, use_cache=True, return_dict=True)
        pkv_f = out_f.past_key_values
        logits_f = out_f.logits[:, -1, :]

        out_v = lm(inputs_embeds=tok_emb, attention_mask=attn_v, past_key_values=pkv_v, use_cache=True, return_dict=True)
        pkv_v = out_v.past_key_values
        logits_v = out_v.logits[:, -1, :]

        with CausalAttentionContext(model, mode=lang_cf_type, layer_group=lang_cf_layers):
            out_l = lm(inputs_embeds=tok_emb, attention_mask=attn_l, past_key_values=pkv_l, use_cache=True, return_dict=True)
        pkv_l = out_l.past_key_values
        logits_l = out_l.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def infer_main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f">>> Running Inference on device: {device}")

    base_output_dir = f"./rpg_outputs/{args.dataset}"
    if args.template_file is None:
        args.template_file = os.path.join(base_output_dir, "templates.json")
    if args.data_path is None:
        args.data_path = os.path.join(base_output_dir, "decomposed_reports.jsonl")

    final_lora_path = "./rpg_llava_lora"
    final_aux_path = os.path.join(final_lora_path, "rpg_aux.pt")

    model = RPGModel(
        lm_name=args.lm_name,
        clip_model_name=args.clip_name,
        template_model_name=args.template_enc_name,
        lora_ckpt=final_lora_path,
        aux_ckpt=final_aux_path,
    )
    model.to(device, dtype=torch.float16)
    model.eval()

    if not os.path.exists(args.template_file):
        raise FileNotFoundError(f"Template file not found: {args.template_file}")
    with open(args.template_file, "r", encoding="utf-8") as f:
        template_list = json.load(f)
    print(f"Loaded {len(template_list)} templates.")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    sample = None
    image_root = f"../datasets/{args.dataset}/images"
    with open(args.data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.idx:
                item = json.loads(line)
                image_path = item["image_path"][0]
                full_img_path = os.path.join(image_root, image_path)
                if not os.path.exists(full_img_path):
                    raise FileNotFoundError(f"Image not found: {full_img_path}")
                image = Image.open(full_img_path).convert("RGB")
                gt_report = item.get("orig_report", "")
                sample = {"image": image, "gt_report": gt_report}
                break
    if sample is None:
        raise IndexError(f"Index {args.idx} out of bounds in {args.data_path}")

    image = sample["image"]
    gt_report = sample["gt_report"]

    dtype = torch.float16
    h_v = model.visual_encoder([image]).to(device, dtype=dtype)
    t_embs = model.template_encoder(template_list).to(device)
    h_t_mean = t_embs.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)

    prefix_f = model.cross_aligner(h_v, h_t_mean).squeeze(1)

    # implicit visual CF prefix
    h_v_cf = model.make_implicit_counterfactual(h_v, mode=args.cf_mode, drop_ratio=args.drop_ratio)
    prefix_cf_v = model.cross_aligner(h_v_cf, h_t_mean).squeeze(1)

    report_text = generate_rpg_causalmm(
        model=model,
        prefix_f=prefix_f,
        prefix_cf_v=prefix_cf_v,
        adapter_name="residual",
        max_new_tokens=args.max_tokens_residual,
        gamma=args.gamma,
        epsilon=args.epsilon,
        temperature=args.temperature,
        top_p=args.top_p,
        lang_cf_type=args.lang_cf_type,
        lang_cf_layers=args.lang_cf_layers,
    )

    print("\n" + "=" * 40)
    print("FINAL RPG REPORT (RPG + CAUSALMM)")
    print("=" * 40)
    print("Findings:")
    print(report_text)
    print("=" * 40)

    if gt_report:
        print("\n===== Ground Truth Report =====")
        print(gt_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="iu_xray")
    parser.add_argument("--template_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--idx", type=int, default=45)

    parser.add_argument("--lm_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--template_enc_name", type=str, default="pritamdeka/S-BioBert-snli-multinli-stsb")

    parser.add_argument("--max_tokens_residual", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")

    # RPG / CAUSALMM params
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    # implicit visual CF
    parser.add_argument("--cf_mode", type=str, default="patch_drop",
                        choices=["patch_drop", "patch_shuffle"])
    parser.add_argument("--drop_ratio", type=float, default=0.3)

    # language attention CF
    parser.add_argument("--lang_cf_type", type=str, default="random",
                        choices=["random", "uniform", "shuffle"])
    parser.add_argument("--lang_cf_layers", type=str, default="middle",
                        choices=["early", "middle", "late", "all"])

    args = parser.parse_args()
    infer_main(args)
