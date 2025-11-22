# RPG/dataset.py

import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

YES_TOKENS = ["yes", "no", "left", "right", "calcification", "benign", "malignant"]

def build_prompt(question: str):
    return f"Question: {question}\nAnswer:"

class VQACollator:
    def __init__(self, tokenizer, image_size=224):
        self.tokenizer = tokenizer
        self.proc = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def img_to_tensor(self, path):
        img = Image.open(path).convert("RGB")
        return self.proc(img)

    def make_counterfactual(self, img: torch.FloatTensor):
        # 随机擦除（反事实）
        x = img.clone()
        C,H,W = x.shape
        for _ in range(2):
            h = random.randint(H//8, H//3)
            w = random.randint(W//8, W//3)
            y0 = random.randint(0, H-h)
            x0 = random.randint(0, W-w)
            x[:, y0:y0+h, x0:x0+w] = 0.0
        return x

    def __call__(self, batch):
        prompts = [build_prompt(b["text"]) for b in batch]
        answers = [b["label"].strip().lower() for b in batch]
        # 仅保留 vocab 中的答案（不在则直接原文）
        answers = [a if a in YES_TOKENS else a for a in answers]

        # 构造两份输入：
        # 1) prior: 只包含问题
        enc_prior = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        # 2) residual/main: 同样的输入（这里简化方式——用同一 prompt；若你想在文本里显式声明“[IMG]”也行）
        enc_main = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

        # teacher forcing：labels = prompt + 空格 + 答案 + eos
        labels_text = [p + " " + a for p,a in zip(prompts, answers)]
        enc_lab = self.tokenizer(labels_text, padding=True, truncation=True, return_tensors="pt")

        # 把 prompt 部分 label 置为 -100，只监督答案 token
        labels = enc_lab.input_ids.clone()
        prompt_len = enc_prior.input_ids.shape[1]
        labels[:, :prompt_len] = -100

        # 图像
        pixel_values = torch.stack([self.img_to_tensor(b["image"]) for b in batch], dim=0)
        pixel_values_cf = torch.stack([self.make_counterfactual(self.img_to_tensor(b["image"])) for b in batch], dim=0)

        return {
            "input_ids_q": enc_prior.input_ids,
            "attention_mask_q": enc_prior.attention_mask,
            "input_ids_qimg": enc_main.input_ids,
            "attention_mask_qimg": enc_main.attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "pixel_values_cf": pixel_values_cf
        }

class VQADataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
        