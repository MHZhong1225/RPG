# infer.py

import random, numpy as np, torch
seed = 77
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
import os
import json
import types

from tqdm import tqdm

from model_rpg import RPGModel
from rpg_dataset import RPGDataset
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
)
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor


from torch.nn import functional as F

import re

from typing import Union, List

def conceptize(t: Union[str, List[str]]):
    """
    1) 输入 str:  返回 concept str
    2) 输入 list[str]: 返回 concept list，并且按归一化后 key 去重(保留原顺序)
    """
    def _conceptize_one(x: str) -> str:
        s = x.lower()

        s = re.sub(r"\b(no|without|negative for|free of|normal|unremarkable|clear|limits|limit|within)\b", "", s)

        s = re.sub(r"\b(there is|there are|is|are|of|the|a|an|to|for|with|and|or)\b", "", s)

        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _norm_key(x: str) -> str:
        s = x.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # -------- case 1: str --------
    if isinstance(t, str):
        return _conceptize_one(t)

    # -------- case 2: list[str] --------
    if isinstance(t, list):
        concepts = [_conceptize_one(x) for x in t]

        seen = set()
        out = []
        for c in concepts:
            k = _norm_key(c)
            if k and k not in seen: 
                out.append(c)
                seen.add(k)
        return out

@torch.no_grad()
def retrieve_topk_templates(model, image, template_list, k=5):
    device = next(model.parameters()).device
    dtype = model.lm.dtype

    # 1) visual features: [1, Nv, Dv]
    h_v = model.visual_encoder([image]).to(device=device, dtype=dtype)
    v_vec = h_v.mean(dim=1)  # [1, Dv]

    # 2) project visual -> hidden space
    v_hid = model.cross_aligner.proj_v(v_vec)  # [1, H]

    # 3) encode templates -> [M, Dt]
    t_embs = model.template_encoder(template_list).to(device=device, dtype=dtype)

    # 4) project templates -> hidden space
    t_hid = model.cross_aligner.proj_t(t_embs)  # [M, H]

    # 5) cosine similarity in hidden space
    v_hid = F.normalize(v_hid, dim=-1)   # [1, H]
    t_hid = F.normalize(t_hid, dim=-1)   # [M, H]

    sims = (t_hid @ v_hid.transpose(0, 1)).squeeze(1)  # [M]
    topk_idx = sims.topk(k).indices.tolist()

    return [template_list[i] for i in topk_idx]

# ... [中间的 CausalAttentionContext 和 generate_rpg_causalmm 代码保持不变] ...

class CausalAttentionContext:
    """
    推理期语言注意力干预(do(A_t = A*_t))。
    只在进入该上下文时 patch LLM 的 self-attn forward，
    退出时恢复原始 forward。
    mode: 'original' | 'random' | 'uniform' | 'shuffle'
    layer_group: 'early' | 'middle' | 'late' | 'all'
    """
    def __init__(self, model, mode="original", scaling_factor=1.0, layer_group="middle"):
        self.model = model.lm
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.layer_group = layer_group
        self.original_forward_funcs = {}
        self.modules_to_patch = []

        layers = None
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "layers"):
            layers = self.model.language_model.model.layers
        else:
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
            else:
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
            attn_output = outputs[0] if isinstance(outputs, tuple) else outputs

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


@torch.no_grad()
def generate_rpg_causalmm(
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
    三分支 RPG / CausalMM 对齐推理：
      logits_f    : factual (真实图像 + 模板 prior)
      logits_cf_v : visual counterfactual(扰动视觉)
      logits_cf_l : language counterfactual(去模板 prior)
    diffs = (1+2gamma) * f - gamma * cf_v - gamma * cf_l
    cutoff = log(ε) + max(f)
    """
    lm = model.lm
    tokenizer = model.tokenizer
    device = prefix_vec_f.device
    input_emb_layer = lm.get_input_embeddings()
    lm.set_adapter(adapter_name)

    # -------- prompt --------
    prompt_text = (
        "You are a radiologist. Write radiology FINDINGS for this CHEST X-ray.\n"
        "**FORMAT RULES**:\n"
        "1) Use a numbered list with at MOST SIX items.\n"
        "2) Each item is ONE sentence, concise, focusing on abnormalities.\n"
        "3) Prioritize abnormal findings; If a structure is normal, you may state negative findings, but use at MOST SIX negative items total. \nFindings: "
    )
    # prompt_text = (
    #     "You are a board-certified radiologist. Write the radiology FINDINGS for this CHEST X-ray.\n"
    #     "Follow these rules strictly:\n"
    #     "1) Output ONLY the FINDINGS (no Impression, no Diagnosis, no headings).\n"
    #     "2) Use a numbered list with at most 7 items.\n"
    #     "3) Each item must be exactly ONE concise sentence.\n"
    #     "4) Prioritize abnormal findings; include normal/negative findings ONLY if clinically relevant.\n"
    #     "5) Use no more than TWO negative/normal items in total.\n"
    #     "6) If there are no abnormalities, you may state a negative finding.\n"
    #     "7) Describe what you see; do NOT speculate on etiology beyond imaging appearance.\n"
    #     "Findings:"
    # )
        # "Only describe thoracic structures: lungs, pleura, mediastinum, heart, chest wall, ribs, visible spine.\n"
        # "Do NOT mention abdomen, pelvis, hip, knee, extremities, or unrelated organs.\n"
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_embeds = input_emb_layer(prompt_ids)
    prompt_mask = torch.ones((1, prompt_ids.size(1)), dtype=torch.long, device=device)

    # BOS
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    bos_ids = torch.tensor([[bos_id]], device=device)
    bos_embeds = input_emb_layer(bos_ids)
    bos_mask = torch.ones((1, 1), dtype=torch.long, device=device)

    # -------- prefix embeds --------
    p_emb_f, p_mask_f = model._build_prefix(prefix_vec_f)
    p_emb_v, p_mask_v = model._build_prefix(prefix_vec_cf_v)
    p_emb_l, p_mask_l = model._build_prefix(prefix_vec_cf_l)

    inputs_embeds_f = torch.cat([p_emb_f, bos_embeds, prompt_embeds], dim=1)
    attn_f = torch.cat([p_mask_f, bos_mask, prompt_mask], dim=1)

    inputs_embeds_v = torch.cat([p_emb_v, bos_embeds, prompt_embeds], dim=1)
    attn_v = torch.cat([p_mask_v, bos_mask, prompt_mask], dim=1)

    inputs_embeds_l = torch.cat([p_emb_l, bos_embeds, prompt_embeds], dim=1)
    attn_l = torch.cat([p_mask_l, bos_mask, prompt_mask], dim=1)

    # -------- logits processors / warpers --------
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        TopPLogitsWarper,
        TemperatureLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        # NoBadWordsLogitsProcessor,
        # NoRepeatNGramLogitsProcessor,
    )

    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(1.2))

    warpers = LogitsProcessorList()
    if temperature is not None and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_p is not None and top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))

    # -------- init forward (build kv cache) --------
    out_f = lm(inputs_embeds=inputs_embeds_f, attention_mask=attn_f, use_cache=True, return_dict=True)
    pkv_f = out_f.past_key_values
    logits_f = out_f.logits[:, -1, :]

    out_v = lm(inputs_embeds=inputs_embeds_v, attention_mask=attn_v, use_cache=True, return_dict=True)
    pkv_v = out_v.past_key_values
    logits_v = out_v.logits[:, -1, :]

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

        final_logits = processors(input_ids_shared, final_logits)
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

        out_l = lm(inputs_embeds=tok_emb, attention_mask=attn_l, past_key_values=pkv_l, use_cache=True, return_dict=True)
        pkv_l = out_l.past_key_values
        logits_l = out_l.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def infer_main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f">>> Running Inference on device: {device}")

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

    base_output_dir = f"./rpg_outputs/{args.dataset}"
    template_file = args.template_file or os.path.join(base_output_dir, "templates.json")
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    with open(template_file, "r", encoding="utf-8") as f:
        template_list = json.load(f)
    def is_bad_template(s):
        s_low = s.lower()
        return ("xxxx" in s_low) or ("___" in s_low) or ("__" in s_low)
    template_list = [t for t in template_list if not is_bad_template(t)]
    print(f"Loaded: {len(template_list)} templates.")

    data_path = args.data_path or os.path.join(base_output_dir, "decomposed_reports.jsonl")
    image_root = args.image_root or f"../datasets/{args.dataset}/images"

    dataset = RPGDataset(jsonl_path=data_path, split=args.split, image_root=image_root)
    if len(dataset) == 0:
        raise ValueError(f"No samples found for split={args.split}. Check jsonl 'split' field.")
    if args.idx < 0 or args.idx >= len(dataset):
        raise IndexError(f"idx={args.idx} out of range for split={args.split} (n={len(dataset)})")

    sample = dataset[args.idx]
    image = sample["images"][0]
    gt_report = sample.get("orig_report", "")

    dtype = torch.float16
    h_v = model.visual_encoder([image]).to(device, dtype=dtype)

    # top-k 
    top_templates = retrieve_topk_templates(model, image, template_list, k=args.topk)
    top_concepts = conceptize(top_templates)
    h_t = model.template_encoder(top_concepts).to(device)
    h_t_mean = h_t.mean(dim=0, keepdim=True).to(dtype=dtype)

    print(f"top_concepts: {top_concepts}")
    # t_embs = model.template_encoder(template_list).to(device)
    # h_t_mean = t_embs.mean(dim=0, keepdim=True).to(device=device, dtype=dtype)

    # factual prefix
    prefix_f = model.cross_aligner(h_v, h_t_mean)    # [1,K,H]  不用 squeeze

    # visual cf prefix
    h_v_cf = model.make_implicit_counterfactual(h_v, mode=args.cf_mode, drop_ratio=args.drop_ratio)
    prefix_cf_v = model.cross_aligner(h_v_cf, h_t_mean)

    # language cf prefix: remove template prior (h_t=0)
    h_t_zero = torch.zeros_like(h_t_mean)
    prefix_cf_l = model.cross_aligner(h_v, h_t_zero)
    print("prefix_f:", prefix_f.shape)
    print("prefix_cf_v:", prefix_cf_v.shape)
    print("prefix_cf_l:", prefix_cf_l.shape)
    # ==================== [新增部分：打印 Top-K 模板] ====================
    print("\n" + "=" * 20)
    print(f"Top-{args.topk} Retrieved Templates:")
    for i, t in enumerate(top_templates):
        print(f"[{i+1}] {t}")
    print("=" * 20 + "\n")
    # ===================================================================
    report_text = generate_rpg_causalmm(
        model=model,
        prefix_vec_f=prefix_f,
        prefix_vec_cf_v=prefix_cf_v,
        prefix_vec_cf_l=prefix_cf_l,
        adapter_name="residual",
        max_new_tokens=args.max_tokens_residual,
        gamma=args.gamma,
        epsilon=args.epsilon,
        temperature=args.temperature,
        top_p=args.top_p,
    )


    print("\n" + "=" * 40)
    print(f"FINAL RPG REPORT (split={args.split}, idx={args.idx})")
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
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="which split to infer on (use train/val if no test)")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)

    parser.add_argument("--template_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)

    parser.add_argument("--lm_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--template_enc_name", type=str, default="pritamdeka/S-BioBert-snli-multinli-stsb")

    parser.add_argument("--max_tokens_residual", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--cf_mode", type=str, default="patch_drop",
                        choices=["patch_drop", "patch_shuffle"])
    parser.add_argument("--drop_ratio", type=float, default=0.3)

    parser.add_argument("--lang_cf_type", type=str, default="random",
                        choices=["random", "uniform", "shuffle"])
    parser.add_argument("--lang_cf_layers", type=str, default="middle",
                        choices=["early", "middle", "late", "all"])

    args = parser.parse_args()
    infer_main(args)