
# train_rpg_corrected.py
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from model_rpg import RPGModel   # <-- use corrected model file
from rpg_dataset import RPGDataset, rpg_collate_fn


# ---------------------------
# helpers
# ---------------------------
def freeze_all_but_adapter(
    model: RPGModel,
    adapter_name: str,
    train_template_proj: bool = True,
    train_cross_aligner: bool = True,
):
    """
    冻结全部参数，仅训练指定 adapter 的 LoRA 参数 +（可选）aux模块。
    兼容 PEFT 不同版本的参数命名：
      - 优先匹配包含 adapter_name 的 LoRA 参数
      - 若未匹配到任何 LoRA 参数，则退化为解冻所有 LoRA 参数
    """
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze LoRA params for this adapter (best effort)
    matched = 0
    for n, p in model.lm.named_parameters():
        if "lora" in n and adapter_name in n:
            p.requires_grad = True
            matched += 1

    # fallback: some peft versions don't include adapter name in param names
    if matched == 0:
        for n, p in model.lm.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    # aux modules
    if train_template_proj:
        for p in model.template_proj.parameters():
            p.requires_grad = True

    if train_cross_aligner:
        for p in model.cross_aligner.parameters():
            p.requires_grad = True


def export_one_adapter(src_root: str, adapter_name: str, dst_dir: str):
    """
    PEFT 多 adapter 保存后常见结构：
      src_root/<adapter_name>/adapter_config.json
    本函数会自动找到真正的 adapter 文件夹，然后只拷贝这一份到 dst_dir。
    """
    cand1 = src_root
    cand2 = os.path.join(src_root, adapter_name)

    if os.path.exists(os.path.join(cand1, "adapter_config.json")):
        real_src = cand1
    elif os.path.exists(os.path.join(cand2, "adapter_config.json")):
        real_src = cand2
    else:
        raise ValueError(
            f"[export_one_adapter] Can't find adapter_config.json in "
            f"{cand1} or {cand2}"
        )

    shutil.rmtree(dst_dir, ignore_errors=True)
    shutil.copytree(real_src, dst_dir)


# ---------------------------
# 2. train / eval
# ---------------------------
def train_template_epoch(model, dataloader, optimizer, device, max_length=128, loss_log=None):
    model.train()
    pbar = tqdm(dataloader)
    for batch in pbar:
        templates = batch["templates"]

        loss_t, _ = model.forward_template(
            template_texts=templates,
            labels_texts=None,
            max_length=max_length,
        )

        optimizer.zero_grad()
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss_log is not None:
            loss_log["template_train"].append(loss_t.item())
        pbar.set_description(f"[Template][train] Loss: {loss_t.item():.4f}")


@torch.no_grad()
def eval_template_epoch(model, dataloader, device, max_length=128, loss_log=None):
    model.eval()
    losses = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        templates = batch["templates"]
        loss_t, _ = model.forward_template(
            template_texts=templates,
            labels_texts=None,
            max_length=max_length,
        )
        losses.append(loss_t.item())
        pbar.set_description(f"[Template][val] Loss: {loss_t.item():.4f}")

    mean_loss = sum(losses) / max(1, len(losses))
    if loss_log is not None:
        loss_log["template_val"].append(mean_loss)
    return mean_loss


def train_residual_epoch(
    model,
    dataloader,
    optimizer,
    device,
    max_length=128,
    lambda_causal=0.1,
    lambda_lang_causal=0.1,
    cf_mode="patch_drop",
    drop_ratio=0.3,
    loss_log=None,
):
    """
    Residual 阶段训练（最终 RPG）：
      factual
      visual implicit CF  -> causal loss
      language CF (h_t=0, drop_ratio=0) -> causal loss
    """
    model.train()
    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch["images"]
        templates = batch["templates"]
        pathologies = batch["pathologies"]

        # ---------- factual ----------
        outputs_f, h_v, h_t, r_p = model.forward_residual(
            images=images,
            template_texts=templates,
            pathology_texts=pathologies,
            max_length=max_length,
            use_labels=True,
            return_hidden=True,
            return_prefix=True,
        )
        loss_f = outputs_f.loss

        # ---------- visual implicit cf ----------
        outputs_cf_v, r_p_cf_v = model.forward_residual_implicit_cf(
            h_v=h_v,
            h_t=h_t,
            pathology_texts=pathologies,
            max_length=max_length,
            use_labels=False,
            return_hidden=True,
            cf_mode=cf_mode,
            drop_ratio=drop_ratio,
            return_prefix=True,
        )

        h_f = outputs_f.hidden_states[-1]
        h_cf_v = outputs_cf_v.hidden_states[-1]
        loss_causal_hidden_v = F.mse_loss(h_f.float(), h_cf_v.float())
        loss_causal_prefix_v = F.mse_loss(r_p.float(), r_p_cf_v.float())
        loss_causal_v = 0.5 * loss_causal_hidden_v + 0.5 * loss_causal_prefix_v

        # ---------- language cf (remove template prior) ----------
        if lambda_lang_causal > 0:
            h_t_zero = torch.zeros_like(h_t)
            outputs_cf_l, r_p_cf_l = model.forward_residual_implicit_cf(
                h_v=h_v,
                h_t=h_t_zero,
                pathology_texts=pathologies,
                max_length=max_length,
                use_labels=False,
                return_hidden=True,
                cf_mode=cf_mode,
                drop_ratio=0.0,   # ensure visual not changed
                return_prefix=True,
            )

            h_cf_l = outputs_cf_l.hidden_states[-1]
            loss_causal_hidden_l = F.mse_loss(h_f.float(), h_cf_l.float())
            loss_causal_prefix_l = F.mse_loss(r_p.float(), r_p_cf_l.float())
            loss_causal_l = 0.5 * loss_causal_hidden_l + 0.5 * loss_causal_prefix_l
        else:
            loss_causal_l = torch.tensor(0.0, device=device)

        loss = loss_f + lambda_causal * loss_causal_v + lambda_lang_causal * loss_causal_l

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss_log is not None:
            loss_log["residual_train_total"].append(loss.item())
            loss_log["residual_train_f"].append(loss_f.item())
            loss_log["residual_train_causal_vis"].append(loss_causal_v.item())
            loss_log["residual_train_causal_lang"].append(loss_causal_l.item())

        pbar.set_description(
            f"[Residual][train] Loss: {loss.item():.4f} "
            f"(f={loss_f.item():.4f}, cv={loss_causal_v.item():.4f}, cl={loss_causal_l.item():.4f})"
        )


@torch.no_grad()
def eval_residual_epoch(
    model,
    dataloader,
    device,
    max_length=128,
    lambda_causal=0.1,
    lambda_lang_causal=0.1,
    cf_mode="patch_drop",
    drop_ratio=0.3,
    loss_log=None,
):
    model.eval()
    losses = []
    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch["images"]
        templates = batch["templates"]
        pathologies = batch["pathologies"]

        outputs_f, h_v, h_t, r_p = model.forward_residual(
            images=images,
            template_texts=templates,
            pathology_texts=pathologies,
            max_length=max_length,
            use_labels=True,
            return_hidden=True,
            return_prefix=True,
        )
        loss_f = outputs_f.loss

        outputs_cf_v, r_p_cf_v = model.forward_residual_implicit_cf(
            h_v=h_v,
            h_t=h_t,
            pathology_texts=pathologies,
            max_length=max_length,
            use_labels=False,
            return_hidden=True,
            cf_mode=cf_mode,
            drop_ratio=drop_ratio,
            return_prefix=True,
        )

        h_f = outputs_f.hidden_states[-1]
        h_cf_v = outputs_cf_v.hidden_states[-1]
        loss_causal_hidden_v = F.mse_loss(h_f.float(), h_cf_v.float())
        loss_causal_prefix_v = F.mse_loss(r_p.float(), r_p_cf_v.float())
        loss_causal_v = 0.5 * loss_causal_hidden_v + 0.5 * loss_causal_prefix_v

        if lambda_lang_causal > 0:
            h_t_zero = torch.zeros_like(h_t)
            outputs_cf_l, r_p_cf_l = model.forward_residual_implicit_cf(
                h_v=h_v,
                h_t=h_t_zero,
                pathology_texts=pathologies,
                max_length=max_length,
                use_labels=False,
                return_hidden=True,
                cf_mode=cf_mode,
                drop_ratio=0.0,
                return_prefix=True,
            )
            h_cf_l = outputs_cf_l.hidden_states[-1]
            loss_causal_hidden_l = F.mse_loss(h_f.float(), h_cf_l.float())
            loss_causal_prefix_l = F.mse_loss(r_p.float(), r_p_cf_l.float())
            loss_causal_l = 0.5 * loss_causal_hidden_l + 0.5 * loss_causal_prefix_l
        else:
            loss_causal_l = torch.tensor(0.0, device=device)

        loss = loss_f + lambda_causal * loss_causal_v + lambda_lang_causal * loss_causal_l
        losses.append(loss.item())

        pbar.set_description(
            f"[Residual][val] Loss: {loss.item():.4f} "
            f"(f={loss_f.item():.4f}, cv={loss_causal_v.item():.4f}, cl={loss_causal_l.item():.4f})"
        )

    mean_loss = sum(losses) / max(1, len(losses))
    if loss_log is not None:
        loss_log["residual_val_total"].append(mean_loss)
    return mean_loss


# ---------------------------
# 3. main
# ---------------------------
def main(args):
    train_jsonl = args.train_jsonl

    batch_size = args.batch_size
    num_workers = args.num_workers
    max_length = args.max_length

    template_epochs = args.t_epochs
    residual_epochs = args.r_epochs

    lambda_causal = args.lambda_causal
    lambda_lang_causal = args.lambda_lang_causal

    cf_mode = args.cf_mode
    drop_ratio = args.dropout

    lr = args.lr
    device = args.device if torch.cuda.is_available() else "cpu"

    loss_log = {
        "template_train": [],
        "template_val": [],
        "residual_train_total": [],
        "residual_train_f": [],
        "residual_train_causal_vis": [],
        "residual_train_causal_lang": [],
        "residual_val_total": [],
    }

    # model
    model = RPGModel(
        lm_name=args.lm_name,
        clip_model_name=args.clip_name,
        template_model_name=args.template_enc_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_prefix_tokens=args.num_prefix_tokens,
        lora_ckpt=None,
        aux_ckpt=None,
    )
    model.to(device)

    # datasets / loaders
    train_dataset = RPGDataset(jsonl_path=train_jsonl, split="train", image_root=args.image_root)
    val_dataset   = RPGDataset(jsonl_path=train_jsonl, split="val", image_root=args.image_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rpg_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rpg_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    os.makedirs("./checkpoints", exist_ok=True)

    # ========= Stage 1 =========
    print("\n===== Stage 1: Train Template Path (LoRA_t) =====")
    model.lm.set_adapter("template")
    # Stage1 不需要训练 cross_aligner（template 路径不走它）
    freeze_all_but_adapter(model, "template", train_template_proj=True, train_cross_aligner=False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_t_val = float("inf")
    best_t_dir = "./checkpoints/best_template_lora"
    best_t_aux = "./checkpoints/best_template_aux.pt"

    for ep in range(template_epochs):
        print(f"Epoch {ep+1}/{template_epochs}")
        train_template_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
            loss_log=loss_log,
        )
        val_loss = eval_template_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            max_length=max_length,
            loss_log=loss_log,
        )
        print(f"[Template] val_loss = {val_loss:.4f}")

        if val_loss < best_t_val:
            best_t_val = val_loss
            model.lm.save_pretrained(
                best_t_dir,
                save_adapter=True,
                adapter_name="template",
            )
            torch.save(
                {
                    "cross_aligner": model.cross_aligner.state_dict(),
                    "template_proj": model.template_proj.state_dict(),
                },
                best_t_aux
            )
            print(f"Saved best TEMPLATE LoRA to {best_t_dir}")
            print(f"Saved best TEMPLATE aux to  {best_t_aux}")

    # ========= Stage 2 =========
    print("\n===== Stage 2: Train Residual Path (LoRA_p) =====")
    model.lm.set_adapter("residual")
    freeze_all_but_adapter(model, "residual", train_template_proj=True, train_cross_aligner=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    best_p_val = float("inf")
    best_p_dir = "./checkpoints/best_residual_lora"
    best_p_aux = "./checkpoints/best_residual_aux.pt"

    for ep in range(residual_epochs):
        print(f"Epoch {ep+1}/{residual_epochs}")
        train_residual_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
            lambda_causal=lambda_causal,
            lambda_lang_causal=lambda_lang_causal,
            cf_mode=cf_mode,
            drop_ratio=drop_ratio,
            loss_log=loss_log,
        )
        val_loss = eval_residual_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
            max_length=max_length,
            lambda_causal=lambda_causal,
            lambda_lang_causal=lambda_lang_causal,
            cf_mode=cf_mode,
            drop_ratio=drop_ratio,
            loss_log=loss_log,
        )
        print(f"[Residual] val_loss = {val_loss:.4f}")

        if val_loss < best_p_val:
            best_p_val = val_loss
            model.lm.save_pretrained(
                best_p_dir,
                save_adapter=True,
                adapter_name="residual",
            )
            torch.save(
                {
                    "cross_aligner": model.cross_aligner.state_dict(),
                    "template_proj": model.template_proj.state_dict(),
                },
                best_p_aux
            )
            print(f"Saved best RESIDUAL LoRA to {best_p_dir}")
            print(f"Saved best RESIDUAL aux to  {best_p_aux}")

    # ========= Export =========
    print("\n===== Export BEST adapters (clean PEFT structure) =====")
    os.makedirs("./rpg_llava_lora", exist_ok=True)

    export_one_adapter(
        src_root=best_t_dir,
        adapter_name="template",
        dst_dir="./rpg_llava_lora/template"
    )
    export_one_adapter(
        src_root=best_p_dir,
        adapter_name="residual",
        dst_dir="./rpg_llava_lora/residual"
    )

    shutil.copy2(best_p_aux, "./rpg_llava_lora/rpg_aux.pt")
    print("BEST template adapter -> ./rpg_llava_lora/template")
    print("BEST residual adapter -> ./rpg_llava_lora/residual")
    print("BEST aux -> ./rpg_llava_lora/rpg_aux.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data / train
    parser.add_argument("--train_jsonl", type=str, default="./rpg_outputs/iu_xray/decomposed_reports.jsonl")
    parser.add_argument("--image_root", type=str, default="../datasets/iu_xray/images")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--t_epochs", type=int, default=25)
    parser.add_argument("--r_epochs", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda:1")

    # model ids
    parser.add_argument("--lm_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--template_enc_name", type=str, default="pritamdeka/S-BioBert-snli-multinli-stsb")

    # lora config
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # prefix config
    parser.add_argument("--num_prefix_tokens", type=int, default=32)

    # causal / RPG losses
    parser.add_argument("--lambda_causal", type=float, default=0.1, help="visual implicit CF causal weight")
    parser.add_argument("--lambda_lang_causal", type=float, default=0.1, help="language CF causal weight (set 0 to disable)")

    # implicit visual CF settings
    parser.add_argument("--cf_mode", type=str, default="patch_drop",
                        choices=["patch_drop", "patch_shuffle"])
    parser.add_argument("--dropout", type=float, default=0.3, help="drop_ratio for implicit visual CF")

    args = parser.parse_args()
    main(args)
