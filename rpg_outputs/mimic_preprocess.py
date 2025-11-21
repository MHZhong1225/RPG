import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer

############################################
# CONFIG
############################################

DATA_PATH = "../datasets/MIMIC-CXR/mimic_train.json"    
OUTPUT_DIR = "./rpg_outputs/mimic"    
os.makedirs(OUTPUT_DIR, exist_ok=True)

SENT_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
LLM_TOKENIZER_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"


############################################
# 1. 句子切分
############################################

def split_into_sentences(text: str) -> List[str]:
    if text is None: 
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    for sep in ["。", "！", "？", ".", "!", "?"]:
        text = text.replace(sep, sep + "\n")
    sents = [s.strip() for s in text.split("\n") if s.strip()]
    return sents


############################################
# 2. 加载 MIMIC JSON（LIST 格式）
############################################

def load_mimic_json(json_path: str):
    """
    mimic_test.json 是一个 list[ sample ]
    每个 item:
      {
        "id": "...",
        "image": "...",
        "report": "..."
      }
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df["report"] = df["report"].fillna("").astype(str)

    print(f"加载 MIMIC 样本数：{len(df)}")
    return df


############################################
# 3. 构建模板库
############################################

def build_template_library(
    reports: List[str],
    sent_model_name=SENT_MODEL_NAME,
    min_len=3,
    max_len=60,
    n_clusters=200,
    min_freq=10
):
    print(">>> Step 3.1: 抽取句子")
    all_sents = []
    for rep in reports:
        all_sents.extend(split_into_sentences(rep))

    print(f"总句子数：{len(all_sents)}")
    filtered = [s for s in all_sents if min_len <= len(s) <= max_len]
    print(f"过滤后句子数：{len(filtered)}")

    counts = Counter(filtered)

    print(">>> Step 3.2: 向量化句子")
    sent_model = SentenceTransformer(sent_model_name)
    embs = sent_model.encode(filtered, batch_size=64, show_progress_bar=True)
    embs = np.asarray(embs)

    print(">>> Step 3.3: 聚类")
    if n_clusters > len(filtered):
        n_clusters = max(2, len(filtered) // 10)
        print(f"n_clusters 自动调整为 {n_clusters}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = kmeans.fit_predict(embs)
    centers = kmeans.cluster_centers_

    # 每个 cluster 找一个代表句
    cluster_map = {i: [] for i in range(n_clusters)}
    for s, cid, e in zip(filtered, cluster_ids, embs):
        cluster_map[cid].append((s, e))

    candidates = []
    for cid, pairs in cluster_map.items():
        if not pairs:
            continue
        center = centers[cid]
        best_s, _ = min(pairs, key=lambda x: np.linalg.norm(x[1] - center))
        candidates.append(best_s)

    print(f"候选模板句：{len(candidates)}")

    # 按频率过滤
    templates = [s for s in candidates if counts[s] >= min_freq]
    templates = sorted(set(templates))
    print(f"最终模板句数：{len(templates)}")

    # 保存
    out = os.path.join(OUTPUT_DIR, "templates.json")
    with open(out, "w") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    print(f"模板库已保存：{out}")

    return templates


############################################
# 4. 模板匹配器
############################################

class TemplateMatcher:
    def __init__(self, templates: List[str], sent_model_name=SENT_MODEL_NAME, sim_threshold=0.80):
        self.templates = templates
        self.model = SentenceTransformer(sent_model_name)
        self.sim_th = sim_threshold

        print(">>> 向量化模板库")
        self.emb_T = self.model.encode(self.templates, batch_size=64, show_progress_bar=True)
        self.emb_T = np.asarray(self.emb_T)

    def match(self, sentence: str):
        """
        返回 (is_template, matched_template)
        """
        if not sentence.strip():
            return False, None

        v = self.model.encode([sentence])[0]
        sims = cosine_similarity(v.reshape(1, -1), self.emb_T)[0]
        idx = np.argmax(sims)
        if sims[idx] >= self.sim_th:
            return True, self.templates[idx]
        return False, None

    def decompose_report(self, report: str):
        """
        报告切成模板句 + 病灶句 (Y_t, Y_p)
        """
        sents = split_into_sentences(report)
        t_sents, p_sents, labels = [], [], []

        for s in sents:
            is_t, matched = self.match(s)
            labels.append({
                "sentence": s,
                "is_template": is_t,
                "matched": matched
            })
            if is_t:
                t_sents.append(s)
            else:
                p_sents.append(s)

        return " ".join(t_sents), " ".join(p_sents), labels


############################################
# 5. token 级标签
############################################

def token_level_labels(report, sent_labels, tokenizer):
    """
    生成 token_labels: 1=模板句token, 0=异常句token
    """
    text = ""
    char_labels = []

    for info in sent_labels:
        s = info["sentence"]
        is_template = info["is_template"]

        if text:
            text += " "
            char_labels.append(0)

        text += s
        char_labels.extend([1 if is_template else 0] * len(s))

    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]

    token_labels = []
    for (start, end) in offsets:
        if end <= start:
            token_labels.append(0)
            continue

        sub = char_labels[start:end]
        if len(sub) == 0:
            token_labels.append(0)
        else:
            token_labels.append(1 if (sum(sub) / len(sub)) >= 0.5 else 0)

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_labels": token_labels
    }


############################################
# 6. 主流程
############################################

def main():
    print(">>> 加载 MIMIC 数据")
    df = load_mimic_json(DATA_PATH)
    reports = df["report"].tolist()

    print(">>> 构建模板库（可替换为加载已有模板）")
    templates = build_template_library(
        reports,
        min_freq=3,    # mimic-test 很少，阈值要降低
        n_clusters=100 # mimic 测试集很小，聚类数也要降低
    )

    print(">>> 初始化模板匹配器")
    matcher = TemplateMatcher(templates)

    print(">>> 加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(LLM_TOKENIZER_NAME, use_fast=True)

    out_path = os.path.join(OUTPUT_DIR, "decomposed_reports.jsonl")
    fout = open(out_path, "w", encoding="utf-8")

    print(">>> 处理每条报告")
    for i, row in df.iterrows():
        rep = row["report"]

        Y_t, Y_p, sent_labels = matcher.decompose_report(rep)
        tok_pack = token_level_labels(rep, sent_labels, tokenizer)

        rec = {
            "id": row["id"],
            "image_path": row["image"],
            "report": rep,
            "template": Y_t,
            "pathology": Y_p,
            "sent_labels": sent_labels,
            "input_ids": tok_pack["input_ids"],
            "attention_mask": tok_pack["attention_mask"],
            "token_labels": tok_pack["token_labels"],
            "tokenized_text": tok_pack["text"]
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (i+1) % 20 == 0:
            print(f"{i+1}/{len(df)} done")

    fout.close()
    print(f"\n>>> 完成！输出已保存：{out_path}")


if __name__ == "__main__":
    main()