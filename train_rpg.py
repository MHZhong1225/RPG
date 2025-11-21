# train_rpg.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from model_rpg import RPGModel
from rpg_dataset import RPGDataset

# 1. Dataset & collate_fn
class RPGDataset_multi(Dataset):
    """
    JSONL 每行示例：
    {
      "id": "...",
      "split": "train" / "val" / "test",
      "image_path": ["xxx/0.png", "xxx/1.png"],
      "template": "...",
      "pathology": "..."
    }
    """
    def __init__(self, jsonl_path, split: str = "train",
                 image_root: str = "../datasets/iu_xray/images"):
        self.samples = []
        self.image_root = image_root
        self.split = split

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_split = item.get("split", "train")
                if split is not None and item_split != split:
                    continue
                self.samples.append(item)

        print(f"[RPGDataset_multi] Loaded {len(self.samples)} samples for split = {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        images = []
        for rel_path in item["image_path"]:
            img = Image.open(os.path.join(self.image_root, rel_path)).convert("RGB")
            images.append(img)

        return {
            "images": images,              # list[PIL]
            "template": item["template"],
            "pathology": item["pathology"],
        }


def rpg_collate_fn(batch):
    images = [b["images"] for b in batch]       # list[list[PIL]]
    templates = [b["template"] for b in batch]
    pathologies = [b["pathology"] for b in batch]

    return {
        "images": images,
        "templates": templates,
        "pathologies": pathologies,
    }


# 2. train / eval
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
    cf_mode="patch_drop",
    drop_ratio=0.3,
    loss_log=None,
):
    model.train()
    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch["images"]
        templates = batch["templates"]
        pathologies = batch["pathologies"]

        # factual
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

        # implicit counterfactual
        outputs_cf, r_p_cf = model.forward_residual_implicit_cf(
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
        h_cf = outputs_cf.hidden_states[-1]
        loss_causal_hidden = F.mse_loss(h_f.float(), h_cf.float())
        loss_causal_prefix = F.mse_loss(r_p.float(), r_p_cf.float())
        loss_causal = 0.5 * loss_causal_hidden + 0.5 * loss_causal_prefix

        loss = loss_f + lambda_causal * loss_causal

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if loss_log is not None:
            loss_log["residual_train_total"].append(loss.item())
            loss_log["residual_train_f"].append(loss_f.item())
            loss_log["residual_train_causal"].append(loss_causal.item())

        pbar.set_description(
            f"[Residual][train] Loss: {loss.item():.4f} (f={loss_f.item():.4f}, c={loss_causal.item():.4f})"
        )


@torch.no_grad()
def eval_residual_epoch(
    model,
    dataloader,
    device,
    max_length=128,
    lambda_causal=0.1,
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

        outputs_cf, r_p_cf = model.forward_residual_implicit_cf(
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
        h_cf = outputs_cf.hidden_states[-1]
        loss_causal_hidden = F.mse_loss(h_f.float(), h_cf.float())
        loss_causal_prefix = F.mse_loss(r_p.float(), r_p_cf.float())
        loss_causal = 0.5 * loss_causal_hidden + 0.5 * loss_causal_prefix

        loss = loss_f + lambda_causal * loss_causal
        losses.append(loss.item())

        pbar.set_description(
            f"[Residual][val] Loss: {loss.item():.4f} (f={loss_f.item():.4f}, c={loss_causal.item():.4f})"
        )

    mean_loss = sum(losses) / max(1, len(losses))
    if loss_log is not None:
        loss_log["residual_val_total"].append(mean_loss)
    return mean_loss


############################################
# 3. main

def main():
    train_jsonl = "./rpg_outputs/iu_xray/decomposed_reports.jsonl"
    lm_name = "llava-hf/llava-v1.6-mistral-7b-hf"
    clip_name = "openai/clip-vit-base-patch32"
    template_enc_name = "pritamdeka/S-BioBert-snli-multinli-stsb"

    batch_size = 4
    num_workers = 4
    max_length = 128

    template_epochs = 10
    residual_epochs = 10
    lambda_causal = 0.1
    cf_mode = "patch_drop"
    drop_ratio = 0.3

    lr = 1e-5
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # logs
    loss_log = {
        "template_train": [],
        "template_val": [],
        "residual_train_total": [],
        "residual_train_f": [],
        "residual_train_causal": [],
        "residual_val_total": [],
    }

    # model
    model = RPGModel(
        lm_name=lm_name,
        clip_model_name=clip_name,
        template_model_name=template_enc_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_ckpt=None,
        aux_ckpt=None,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # datasets / loaders
    train_dataset = RPGDataset(jsonl_path=train_jsonl, split="train")
    val_dataset   = RPGDataset(jsonl_path=train_jsonl, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=rpg_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=rpg_collate_fn,
    )

    os.makedirs("./checkpoints", exist_ok=True)

    # ========= Stage 1 =========
    print("\n===== Stage 1: Train Template Path (LoRA_t) =====")
    best_t_val = float("inf")
    best_t_path = "./checkpoints/best_template.pt"

    for epoch in range(template_epochs):
        print(f"Epoch {epoch+1}/{template_epochs}")
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
            torch.save(
                {
                    "lm": model.lm.state_dict(),  # LoRA 权重在 lm 里
                    "cross_aligner": model.cross_aligner.state_dict(),
                    "template_proj": model.template_proj.state_dict(),
                },
                best_t_path
            )
            print(f"Saved best template checkpoint to {best_t_path}")

    # ========= Stage 2 =========
    print("\n===== Stage 2: Train Residual Path with Implicit Counterfactual (LoRA_p) =====")
    best_p_val = float("inf")
    best_p_path = "./checkpoints/best_residual.pt"

    for epoch in range(residual_epochs):
        print(f"Epoch {epoch+1}/{residual_epochs}")
        train_residual_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
            lambda_causal=lambda_causal,
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
            cf_mode=cf_mode,
            drop_ratio=drop_ratio,
            loss_log=loss_log,
        )
        print(f"[Residual] val_loss = {val_loss:.4f}")

        if val_loss < best_p_val:
            best_p_val = val_loss
            torch.save(
                {
                    "lm": model.lm.state_dict(),
                    "cross_aligner": model.cross_aligner.state_dict(),
                    "template_proj": model.template_proj.state_dict(),
                },
                best_p_path
            )
            print(f"Saved best residual checkpoint to {best_p_path}")

    # ========= Load best residual then export =========
    print("\nLoading best residual checkpoint before final export...")
    ckpt = torch.load(best_p_path, map_location="cpu")
    model.lm.load_state_dict(ckpt["lm"], strict=False)
    model.cross_aligner.load_state_dict(ckpt["cross_aligner"])
    model.template_proj.load_state_dict(ckpt["template_proj"])
    model.to(device)

    # ========= 保存权重 =========
    print("\nSaving LoRA weights (best) ...")

    model.lm.save_pretrained(
        "./rpg_llava_lora/",
        save_adapter=True,
        adapter_name="template",
    )
    model.lm.save_pretrained(
        "./rpg_llava_lora/",
        save_adapter=True,
        adapter_name="residual",
    )

    torch.save(
        {
            "cross_aligner": model.cross_aligner.state_dict(),
            "template_proj": model.template_proj.state_dict(),
        },
        "./rpg_llava_lora/rpg_aux.pt",
    )

    print("Saved BEST LoRA to ./rpg_llava_lora/")
    print("Saved BEST aux to ./rpg_aux.pt")


if __name__ == "__main__":
    main()