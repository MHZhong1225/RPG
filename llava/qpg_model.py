# RPG/llava/rpg_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from .model.builder import load_pretrained_model 

def _get_model_name_from_path(model_path: str) -> str:
    model_path = model_path.strip("/")
    parts = model_path.split("/")
    if parts[-1].startswith("checkpoint-"):
        return parts[-2] + "_" + parts[-1]
    return parts[-1]
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer

# --- 1. 配置 ---
class RPGConfig:
    def __init__(
        self,
        llm_name: str = "liuhaotian/llava-v1.5-7b", 
        #  (用于 h_t)
        template_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        
        # 用于对齐 h_t 和 h_v 
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
    """
    RPG: Residual Pathology Generator
    """
    def __init__(self, cfg: RPGConfig):
        super().__init__()
        self.cfg = cfg
        self.target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ---  LLaVA  (视觉 + 语言) ---
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        for param in self.llm.parameters():
            param.requires_grad = False
            
        self.hidden_size = self.llm.config.hidden_size
        self.vocab_size = self.llm.config.vocab_size

        self._inject_dual_lora()

        self.template_encoder = SentenceTransformer(
            cfg.template_encoder_name, 
            device=self.target_device
        )
        for param in self.template_encoder.parameters():
            param.requires_grad = False
        
        template_dim = self.template_encoder.get_sentence_embedding_dimension()

        # --- 投影层 ---
        # LoRA_t 从 h_t 生成, LoRA_p 从 h_cross 生成
        
        self.template_projector = nn.Linear(template_dim, self.hidden_size)
        
        # LoRA_t (模板) 只看 h_t 投影
        # LoRA_p (残差) 看 h_t 投影 + 图像特征 (h_v) + 文本 (input_ids)
        
        self.template_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.residual_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        self.template_head.weight.data.copy_(self.llm.lm_head.weight.data)
        self.residual_head.weight.data.copy_(self.llm.lm_head.weight.data)
        
        # --- 因果损失队列  ---
        self.register_buffer("ace_residual_queue", torch.randn(256, self.hidden_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # ... (需要实现队列更新逻辑) TODO...

        self.to(self.target_device)


    def _inject_dual_lora(self):
        # 两个独立的 LoRA 适配器
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 建议 Q/K/V + FFN

        # 1. LoRA_t
        config_t = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        # 2. 残差 LoRA_p
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
        template_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        辅助函数：构建 LLaVA 的输入嵌入
        这需要适配 LLaVA 的 prepare_inputs_for_multimodal 逻辑 TODO
        """
        
        # 1. 获取文本 token 嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. 如果有图像 (h_v)
        if images is not None:
            # LLaVA 逻辑：编码图像并通过 projector
            # image_features = self.llm.get_model().vision_tower(images)
            # image_features = self.llm.get_model().mm_projector(image_features)
            
            # (实际 LLaVA 已在 .forward() 中处理)
            # 我们需要手动将 <image> 占位符替换为 image_features

            pass

        # 3. 如果有模板 (h_t)
        if template_embeds is not None:
            # 投影 h_t
            projected_h_t = self.template_projector(template_embeds)
            # [B, 1, H]
            projected_h_t = projected_h_t.unsqueeze(1) 

            print("projected_h_t:", projected_h_t.shape)
            print("text_embeds:", text_embeds.shape)
            text_embeds = torch.cat([projected_h_t, text_embeds], dim=1)

        return text_embeds
        

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        template_texts: list, # 模板 $P_l$ (纯文本)
        labels_t: torch.LongTensor, # 模板目标 $Y_t^*$
        labels_p: torch.LongTensor, # 异常目标 $Y_p^*$
        pixel_values: Optional[torch.FloatTensor] = None, # 影像 $X_v$
        pixel_values_cf: Optional[torch.FloatTensor] = None, # 反事实影像 $X_v^{cf}$
    ) -> Dict[str, Any]:
        
        # --- 0. h_t (模板先验) ---
        with torch.no_grad():
            h_t = self.template_encoder.encode(
                template_texts, 
                convert_to_tensor=True, 
                device=self.target_device
            )
        # [B, H_template]
        
        # --- 1. 模板分支 (LoRA_t) ---
        #  R_t = LoRA_t(h_t)
        # 我们用 h_t 作为 LLaVA 的输入 (替换图像)
        
        # LLaVA 的 forward 需要 input_ids, 我们用一个简单的 prompt
        # (TODO)
        # 简化：假设 LLaVA 的 forward 可以接受 inputs_embeds
        
        embeds_t = self._get_llm_input_embeds(
            input_ids=input_ids, # 假设 input_ids 包含模板生成的 prompt
            template_embeds=h_t
        )
        
        # 启用 LoRA_t
        self.llm.set_adapter("template")
        out_t = self.llm(
            inputs_embeds=embeds_t,
            attention_mask=attention_mask, # 需调整以匹配 embeds_t
            output_hidden_states=True,
            return_dict=True
        )
        hidden_t = out_t.hidden_states[-1]
        logits_t = self.template_head(hidden_t)
        
        # --- 2. 残差分支 (LoRA_p) ---
        # R_p = LoRA_p(h_cross)
        # 我们用 h_t + X_v (图像) 作为 LLaVA 的输入
        
        # 启用 LoRA_p
        self.llm.set_adapter("residual")
        
        # (A) 事实
        out_p = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=pixel_values, # LLaVA 会自动处理
            # 模板嵌入也应传入 (需要修改 LLaVA 的 forward)
            # template_embeds=h_t,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_p = out_p.hidden_states[-1]
        logits_p = self.residual_head(hidden_p)

        # (B) 反事实
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

        # --- 3. loss ---
        loss_temp = loss_path = loss_causal = loss_perp = torch.tensor(0.0, device=self.target_device)

        # L_temp 
        sl_t, slbl_t = self._shift_for_ce(logits_t, labels_t)
        loss_temp = F.cross_entropy(sl_t.view(-1, self.vocab_size), slbl_t.view(-1), ignore_index=-100)

        # L_path
        sl_p, slbl_p = self._shift_for_ce(logits_p, labels_p)
        loss_path = F.cross_entropy(sl_p.view(-1, self.vocab_size), slbl_p.view(-1), ignore_index=-100)
        
        # L_causal 
        if hidden_p_cf is not None:
            # 截断到相同长度 
            len_p = sl_p.shape[1]
            delta_R = hidden_p[:, :len_p, :] - hidden_p_cf[:, :len_p, :]
            
            # L_causal = ||\Delta R_p - stopgrad(ACE)||
            # 简化 ACE 估计 (使用 batch mean 建议 EMA)
            ace_estimate = torch.mean(delta_R, dim=(0, 1), keepdim=True)
            
            loss_causal = F.mse_loss(delta_R, ace_estimate.detach())

        # L_perp 
        # 在 token 级别计算 Cosine 相似度
        len_perp = min(hidden_t.shape[1], hidden_p.shape[1])
        cos_sim = F.cosine_similarity(hidden_t[:, :len_perp, :], hidden_p[:, :len_perp, :], dim=-1)
        loss_perp = torch.mean(cos_sim) # 我们希望 CosSim 接近 0

        # (lambda 权重应配置)
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
        