# RPG/llava/rpg_model.py

import torch
import torch.nn as nn
# ... (其他 imports)
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model.builder import load_pretrained_model 
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
# 移除 sentence_transformers
import torch.nn.functional as F

from typing import Optional, Dict, Any, Tuple

# 导入 LLaVA 的特定类，确保 vision_tower 和 projector 被正确加载

def _get_model_name_from_path(model_path: str) -> str:
    model_path = model_path.strip("/")
    parts = model_path.split("/")
    if parts[-1].startswith("checkpoint-"):
        return parts[-2] + "_" + parts[-1]
    return parts[-1]
from sentence_transformers import SentenceTransformer


class RPGConfig:
    def __init__(
        self,
        # 推荐使用 LLaVA-Med 或专用的 Med-MLLM
        llm_name: str = "liuhaotian/llava-v1.5-7b", 
        # 模板编码器 (用于 h_t)
        template_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        
        # 用于对齐 h_t 和 h_v (如果需要)
        proj_dim: int = 1024,
    ):
        self.llm_name = llm_name
        self.template_encoder_name = template_encoder_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.proj_dim = proj_dim

# --- 2. RPG 模型 ---
class RPGModel(nn.Module):
    def __init__(self, cfg: RPGConfig):
        super().__init__()
        self.cfg = cfg
        self.target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # --- 加载 LLaVA 主干 (视觉 + 语言) ---
        # ... (这部分加载代码保持不变) ...
        model_name = _get_model_name_from_path(cfg.llm_name)
        self.tokenizer, self.llm, self.image_processor, _ = load_pretrained_model(
            model_path=cfg.llm_name,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # ... (冻结 LLM 和设置 vocab_size 保持不变) ...
        self.hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        # --- 注入双 LoRA 适配器 ---
        self._inject_dual_lora()


        # all-MiniLM-L6-v2 的维度是 384
        template_dim = 384 

        # --- 投影层 ---
        # 1. 模板投影器 (h_t -> hidden_size)
        self.template_projector = nn.Linear(template_dim, self.hidden_size)
        

        self.template_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.residual_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.template_head.weight.data.copy_(self.llm.lm_head.weight.data)
        self.residual_head.weight.data.copy_(self.llm.lm_head.weight.data)
        self.to(self.target_device)

        self._inject_dual_lora()

    def _inject_dual_lora(self):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 方案 §6.2 建议 Q/K/V + FFN

        # 1. 模板分支 LoRA_t
        config_t = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        # 2. 残差分支 LoRA_p
        config_p = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )

        if not isinstance(self.llm, PeftModel):
             self.llm = get_peft_model(self.llm, config_t, adapter_name="template")
        else:
             # peft>=0.8 uses (adapter_name, peft_config)
             self.llm.add_adapter(adapter_name="template", peft_config=config_t)
             
        self.llm.add_adapter(adapter_name="residual", peft_config=config_p)

        self.llm.disable_adapter_layers()
        self.llm.print_trainable_parameters()

    def _get_llm_input_embeds(
        self, 
        input_ids: torch.LongTensor, 
        images: Optional[torch.FloatTensor] = None,
        template_embeds: Optional[torch.FloatTensor] = None, # 接收的是张量
    ) -> torch.FloatTensor:
        # ... (获取 text_embeds 保持不变) ...
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3. 如果有模板 (h_t)
        if template_embeds is not None:
            # 投影 h_t (template_embeds 已经是 [B, H_template])
            projected_h_t = self.template_projector(template_embeds)
            # [B, 1, H]
            projected_h_t = projected_h_t.unsqueeze(1) 

            # !! 移除调试打印 !!
            # print("projected_h_t:", projected_h_t.shape)
            # print("text_embeds:", text_embeds.shape)
            text_embeds = torch.cat([projected_h_t, text_embeds], dim=1)

        return text_embeds
        

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        # !! 签名变更：从 list 变为 Tensor !!
        template_embeds: torch.FloatTensor, # 模板 $P_l$ (已编码)
        labels_t: torch.LongTensor, 
        labels_p: torch.LongTensor, 
        pixel_values: Optional[torch.FloatTensor] = None, 
        pixel_values_cf: Optional[torch.FloatTensor] = None,
    ) -> Dict[str, Any]:
        
        # --- 0. 准备 h_t (已在 collator 中完成) ---
        # h_t = template_embeds (已经是 [B, H_template])
        h_t = template_embeds
        
        # --- 1. 模板分支 (LoRA_t) ---
        embeds_t = self._get_llm_input_embeds(
            input_ids=input_ids, 
            template_embeds=h_t
        )
        
        # !! 修复 attention_mask !!
        # 我们在前面加了一个 token (h_t)，所以 attention_mask 也要加一个 1
        mask_prefix = torch.ones((attention_mask.shape[0], 1), 
                                 dtype=attention_mask.dtype, 
                                 device=attention_mask.device)
        attention_mask_t = torch.cat([mask_prefix, attention_mask], dim=1)
        
        # 启用 LoRA_t
        self.llm.set_adapter("template")
        out_t = self.llm(
            inputs_embeds=embeds_t,
            attention_mask=attention_mask_t, # 使用调整后的 mask
            output_hidden_states=True,
            return_dict=True
        )
        hidden_t = out_t.hidden_states[-1]
        logits_t = self.template_head(hidden_t)
        
        # --- 2. 残差分支 (LoRA_p) ---
        # ... (这部分保持不变) ...
        # (A) 事实
        self.llm.set_adapter("residual")
        out_p = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask, # 残差分支使用原始 mask
            images=pixel_values, 
            output_hidden_states=True,
            return_dict=True
        )
        hidden_p = out_p.hidden_states[-1]
        logits_p = self.residual_head(hidden_p)

        # (B) 反事实 (方案 §4.1)
        hidden_p_cf = None
        if pixel_values_cf is not None:
            out_p_cf = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=pixel_values_cf, # 使用 $X_v^{cf}$
                # template_embeds=h_t,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_p_cf = out_p_cf.hidden_states[-1]

        # --- 3. 损失计算 (方案 §5) ---
        loss_temp = loss_path = loss_causal = loss_perp = torch.tensor(0.0, device=self.target_device)

        # L_temp (方案 §5)
        sl_t, slbl_t = self._shift_for_ce(logits_t, labels_t)
        loss_temp = F.cross_entropy(sl_t.view(-1, self.vocab_size), slbl_t.view(-1), ignore_index=-100)

        # L_path (方案 §5)
        sl_p, slbl_p = self._shift_for_ce(logits_p, labels_p)
        loss_path = F.cross_entropy(sl_p.view(-1, self.vocab_size), slbl_p.view(-1), ignore_index=-100)
        
        # L_causal (方案 §4.1)
        if hidden_p_cf is not None:
            # 截断到相同长度 (CE 移位后)
            len_p = sl_p.shape[1]
            delta_R = hidden_p[:, :len_p, :] - hidden_p_cf[:, :len_p, :]
            
            # 方案 §4.1: L_causal = ||\Delta R_p - stopgrad(ACE)||
            # 简化 ACE 估计 (使用 batch mean，方案 §C 建议 EMA)
            ace_estimate = torch.mean(delta_R, dim=(0, 1), keepdim=True)
            
            loss_causal = F.mse_loss(delta_R, ace_estimate.detach())

        # L_perp (方案 §4.2)
        # 在 token 级别计算 Cosine 相似度
        len_perp = min(hidden_t.shape[1], hidden_p.shape[1])
        cos_sim = F.cosine_similarity(hidden_t[:, :len_perp, :], hidden_p[:, :len_perp, :], dim=-1)
        loss_perp = torch.mean(cos_sim) # 我们希望 CosSim 接近 0

        # 总损失 (方案 §5)
        # (lambda 权重应来自配置)
        lambda_c = 1.0 
        lambda_d = 0.1
        loss = loss_temp + loss_path + lambda_c * loss_causal + lambda_d * loss_perp

        return {
            "loss": loss,
            "loss_temp": loss_temp,
            "loss_path": loss_path,
            "loss_causal": loss_causal,
            "loss_perp": loss_perp,
            "logits_t": logits_t,
            "logits_p": logits_p,
        }

    def _shift_for_ce(self, logits, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return shift_logits, shift_labels