import json
import os
from PIL import Image
from torch.utils.data import Dataset


class RPGDataset(Dataset):
    """
    JSONL 每行示例：
    {
      "id": "CXR1860_IM-0558",
      "split": "train" / "val" / "test",
      "image_path": ["CXR1860_IM-0558/0.png"],
      "template": "...",
      "pathology": "..."
    }
    """

    def __init__(self, jsonl_path, split: str = "train",
                 image_root: str = "../datasets/iu_xray/images"):
        self.samples = []
        self.image_root = image_root
        self.split = split  # 保存用户指定的 split

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # json 中的 split（如果没有，就当作 train）
                item_split = item.get("split", "train")
                if split is not None and item_split != split:
                    continue

                self.samples.append(item)

        print(f"[RPGDataset] Loaded {len(self.samples)} samples for split = {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        img_rel_path = item["image_path"][0]
        images = Image.open(os.path.join(self.image_root, img_rel_path)).convert("RGB")

        return {
            "images": [images],
            "template": item["template"],
            "pathology": item["pathology"],
            "orig_report": item["orig_report"],
        }

def rpg_collate_fn(batch):
    images = [item["images"] for item in batch]
    templates = [item["template"] for item in batch]
    pathologies = [item["pathology"] for item in batch]

    return {
        "images": images,
        "templates": templates,
        "pathologies": pathologies,
    }


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

