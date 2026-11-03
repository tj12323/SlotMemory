# model/memory_compressor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from slotcontrast.modules.groupers import SlotAttention
from slotcontrast.modules.initializers import RandomInit,FixedLearnedInit
from typing import Dict, Optional,Tuple
from torch.utils.checkpoint import checkpoint
from utils.loss import Slot_Slot_Contrastive_Loss
import math

class MemoryCompressor(nn.Module):
    """可微的 Slot-based KV Bank 压缩器（端到端联合训练专用）"""
    
    def __init__(self, 
                 input_dim: int = 1536, 
                 slot_dim: int = 64,
                 num_slots_per_chunk: int = 7,
                 tokens_per_slot: int = 64,
                 trainable: bool = True,
                 freeze_grouper: bool = False):
        super().__init__()
        
        hidden_dim = 4096  # 与 Wan dim 适配（可根据实际 dim 调整）
        
        # 1. Input Adapter（特征适配）
        self.slot_input_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 注意：这里 dim 需与 CausalWanModel.dim 保持一致
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
        # 2. Slot Grouper（核心 SlotAttention）
        self.slot_grouper = SlotAttention(
            inp_dim=slot_dim,
            slot_dim=slot_dim,
            kvq_dim=slot_dim,
            n_iters=2,
            eps=1e-6,
            use_gru=True
        )
        
        # 3. Output Adapter（可选：若未来需要 slot → feature 重投影）
        decoder_hidden = 1024
        self.slot_output_adapter = nn.Sequential(
            nn.Linear(slot_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, input_dim)  # 输出回 Wan dim
        )
        
        # 4. Initializer
        self.slot_initializer = FixedLearnedInit(n_slots=num_slots_per_chunk, dim=slot_dim)
        
        self.num_slots_per_chunk = num_slots_per_chunk
        self.tokens_per_slot = tokens_per_slot
        self.slot_dim = slot_dim
        
        # 5. Dynamics Predictor（方案 2：只新增这一项）
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.LayerNorm(slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim)
        )
        
        # 新增：显式阶段状态，不再靠 requires_grad 猜
        self.stage_mode = "stage1" if freeze_grouper else "stage2"
        self.enable_dynamics_loss = not freeze_grouper
        
        # 可训练开关（端到端关键）
        self.trainable = trainable
        self._set_trainable(trainable, freeze_grouper)
        
        # 1. 实例化 Loss 函数
        self.slot_loss_fn = Slot_Slot_Contrastive_Loss(temperature=0.1, batch_contrast=True)
        self.recon_loss_weight = 0.01
        # ---- slot_output_adapter reconstruction / token selection ----
        self.selection_mask_log_weight = 0.3
        self.selection_eps = 1e-6
        
        self.accumulated_loss = 0.0
        self.loss_count = 0

        self.accumulated_con_loss = 0.0
        self.accumulated_recon_loss = 0.0
        self.accumulated_dyn_loss = 0.0
        self.accumulated_prev_hit = 0.0
        self.distill_count = 0
        
        # 2. 【核心】内置双缓冲状态机 (key: layer_idx)
        self.prev_clean_slots = {}     # 存放 Chunk N-1 的历史
        self.current_clean_slots = {}  # 存放 Chunk N 的暂存
        self.slot_write_scores = {}
        
        self.last_loss_breakdown = {
            "slot_con": 0.0,
            "slot_recon": 0.0,
            "slot_dyn": 0.0,
            "slot_prev_hit": 0.0,
            "slot_stage2": 0.0,
        }
        
        
        # ==================== Diagnostic Logging ====================
        self.enable_diagnostics = False
        self.diagnostic_interval = 20   # 每多少次 forward 打印一次
        self.diagnostic_step = 0

        self.last_diagnostics = {}

        self.accumulated_diag = {
            "slot_entropy": 0.0,
            "slot_top1_conf": 0.0,
            "slot_util_min": 0.0,
            "slot_util_max": 0.0,
            "slot_util_std": 0.0,
            "slot_overlap": 0.0,
            "valid_token_ratio": 0.0,
        }
        self.diag_count = 0
        
        self.use_global_topk = False
        self.use_global_selection = False
        
        # ---- 面积加权重建损失参数（AW-MSE风格） ----
        self.aw_power_p = 0.25        # 面积倒数幂次
        self.aw_cap_w = 5.0         # 权重上限
        self.aw_norm_mode = "mean1"  # 归一化方式
        self.aw_eps = 1e-8
        
        # future-query selection distillation
        self.selection_distill_weight = 0.01  # 可调
        self.selection_temperature = 0.2
        self.con_loss_weight = 10.0
    
    def _set_trainable(self, trainable: bool, freeze_grouper: bool):
        for module in [
            self.slot_input_adapter,
            self.slot_grouper,
            self.slot_output_adapter,
            self.slot_initializer,
            self.dynamics_predictor,
        ]:
            for param in module.parameters():
                param.requires_grad = trainable

        if trainable and freeze_grouper:
            for module in [self.slot_grouper, self.slot_initializer, self.dynamics_predictor]:
                for param in module.parameters():
                    param.requires_grad = False

        self.stage_mode = "stage1" if freeze_grouper else "stage2"
        self.enable_dynamics_loss = not freeze_grouper
    
    @torch.no_grad()
    def compute_slot_diagnostics(
        self,
        masks: torch.Tensor,
        best_indices: torch.Tensor,
        recon_valid: torch.Tensor,
        layer_idx: int,
    ):
        """
        masks: [B, S, L]
        best_indices: [B, S, K]
        recon_valid: [B, L]
        """

        B, S, L = masks.shape
        K = best_indices.shape[-1]

        eps = 1e-8

        # 1. slot mask entropy
        # 对每个 token 看 slot 分布是否尖锐
        prob = masks.clamp_min(eps)
        entropy = -(prob * prob.log()).sum(dim=1)  # [B, L]
        entropy_norm = entropy / math.log(S + eps)
        mean_entropy = entropy_norm.mean().item()

        # 2. top-1 confidence
        top1_conf = masks.max(dim=1).values  # [B, L]
        mean_top1_conf = top1_conf.mean().item()

        # 3. slot utilization
        # 每个 slot 得到多少软分配质量
        slot_mass = masks.sum(dim=-1)  # [B, S]
        slot_mass_norm = slot_mass / (slot_mass.sum(dim=1, keepdim=True) + eps)

        util_min = slot_mass_norm.min(dim=1).values.mean().item()
        util_max = slot_mass_norm.max(dim=1).values.mean().item()
        util_std = slot_mass_norm.std(dim=1).mean().item()

        # 4. top-k overlap between slots
        # 统计不同 slot 选到相同 token 的比例
        overlap_list = []

        for b in range(B):
            slot_sets = []
            for s in range(S):
                slot_sets.append(set(best_indices[b, s].detach().cpu().tolist()))

            pair_overlaps = []
            for i in range(S):
                for j in range(i + 1, S):
                    inter = len(slot_sets[i].intersection(slot_sets[j]))
                    union = len(slot_sets[i].union(slot_sets[j])) + eps
                    pair_overlaps.append(inter / union)

            if len(pair_overlaps) > 0:
                overlap_list.append(sum(pair_overlaps) / len(pair_overlaps))

        mean_overlap = float(sum(overlap_list) / max(len(overlap_list), 1))

        # 5. valid token ratio
        valid_token_ratio = recon_valid.float().mean().item()

        diag = {
            "layer_idx": layer_idx,
            "slot_entropy": mean_entropy,
            "slot_top1_conf": mean_top1_conf,
            "slot_util_min": util_min,
            "slot_util_max": util_max,
            "slot_util_std": util_std,
            "slot_overlap": mean_overlap,
            "valid_token_ratio": valid_token_ratio,
        }

        return diag
    
    
    def set_stage_mode(self, stage_mode: str):
        if stage_mode not in {"stage1", "stage2"}:
            raise ValueError(f"Invalid stage_mode: {stage_mode}")

        self.stage_mode = stage_mode
        self.enable_dynamics_loss = (stage_mode == "stage2")

        if stage_mode == "stage1":
            for module in [self.slot_grouper, self.slot_initializer, self.dynamics_predictor]:
                for param in module.parameters():
                    param.requires_grad = False
            for param in self.slot_input_adapter.parameters():
                param.requires_grad = True
            for param in self.slot_output_adapter.parameters():
                param.requires_grad = True
            print("❄️ [Stage 1] Slot Grouper + Dynamics Predictor are FROZEN. Contrastive-only training.")
        else:
            for module in [
                self.slot_input_adapter,
                self.slot_grouper,
                self.slot_output_adapter,
                self.slot_initializer,
                self.dynamics_predictor,
            ]:
                for param in module.parameters():
                    param.requires_grad = True
            print("🔥 [Stage 2] Slot Grouper + Dynamics Predictor are UNFROZEN. Contrastive-only training.")
        
    def load_pretrained_slot_contrast(self, ckpt_path: str, strict: bool = False):
        """
        专门用于加载 SlotContrast 预训练权重的工具函数
        """
        print(f"Loading pretrained SlotContrast weights from {ckpt_path}...")
        device = next(self.parameters()).device
        ckpt = torch.load(ckpt_path, map_location=device)
        
        # 兼容不同保存格式（有时权重在 'state_dict' 或 'model' 键下）
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
        
        new_state_dict = {}
        matched_keys = []
        
        # 遍历 checkpoint 中的权重键名
        for k, v in state_dict.items():
            # 因为 ckpt 里的 key 是裸奔的（如 to_k.weight），我们直接强行加上前缀
            new_key = f"slot_grouper.{k}"
            
            # 检查拼装后的 key 是否存在于当前模型中
            if new_key in self.state_dict():
                # 检查维度是否完全一致
                if self.state_dict()[new_key].shape == v.shape:
                    new_state_dict[new_key] = v
                    matched_keys.append(new_key)
                else:
                    print(f"⚠️ 维度不匹配跳过: {new_key} (ckpt: {v.shape}, model: {self.state_dict()[new_key].shape})")
            else:
                # 预留打印，以防有漏网之鱼
                print(f"⚠️ 当前模型不需要此参数: {k}")

        # 执行加载
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=strict)
        
        print(f"✅ 成功加载了 {len(matched_keys)} 个预训练张量。")
        if strict or len(matched_keys) == 0:
            print(f"Missing keys (Expected, adapter 应该是 missing 的): {missing_keys[:5]}...")
    
    @staticmethod
    def select_tokens_by_masks(masks, total_keep, use_global_selection=False):
        B, S, L = masks.shape
        if use_global_selection:
            token_scores = masks.max(dim=1).values  # [B, L]
            k = min(total_keep, L)
            _, idx = torch.topk(token_scores, k=k, dim=1)
            if k < total_keep:
                pad = torch.zeros(B, total_keep - k, dtype=torch.long, device=masks.device)
                idx = torch.cat([idx, pad], dim=1)
        else:
            _, idx = torch.topk(masks, k=total_keep // S, dim=2)  # per-slot
            idx = idx.view(B, -1)  # [B, total_keep]
        return idx
    
    def select_tokens_by_reconstruction(
        self,
        features_k: torch.Tensor,
        slots: torch.Tensor,
        masks: torch.Tensor,
        tokens_per_slot: int,
        selection_score: torch.Tensor,      # [B, S, L]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 reconstruction score + soft mask prior 选择每个 slot 的代表 tokens。

        selection_score 越大越应该被选中。
        不再使用 INF hard mask，避免 soft mask 完全压倒 reconstruction error。
        """
        B, L, HD = features_k.shape
        S = slots.shape[1]
        device = features_k.device

        total_keep = S * tokens_per_slot

        k = min(tokens_per_slot, L)
        if k > 0:
            _, best_indices = torch.topk(
                selection_score,
                k=k,
                dim=2,
            )  # [B, S, k]
        else:
            best_indices = torch.zeros(
                B, S, 0,
                dtype=torch.long,
                device=device,
            )

        best_indices_1d = best_indices.reshape(B, -1)

        # 理论上 L_new=1560, S*R=448，不会不够。
        # 这里保留防御式 padding。
        if best_indices_1d.shape[1] < total_keep:
            need = total_keep - best_indices_1d.shape[1]

            global_scores = selection_score.max(dim=1).values  # [B, L]
            _, add_indices = torch.topk(
                global_scores,
                k=min(need, L),
                dim=1,
            )

            if add_indices.shape[1] < need:
                pad = torch.zeros(
                    B,
                    need - add_indices.shape[1],
                    dtype=torch.long,
                    device=device,
                )
                add_indices = torch.cat([add_indices, pad], dim=1)

            best_indices_1d = torch.cat(
                [best_indices_1d, add_indices[:, :need]],
                dim=1,
            )

        return best_indices_1d[:, :total_keep], best_indices

    def forward(self, new_k: torch.Tensor, new_v: torch.Tensor,
            layer_idx: int, query: Optional[torch.Tensor] = None):
        B, L_new, H, D = new_k.shape
        device = new_k.device

        features_k = new_k.flatten(2, 3)
        features_v = new_v.flatten(2, 3)

        # ========== 消融分支：全局随机 Top-K ==========
        if self.use_global_topk:
            total_to_keep = self.num_slots_per_chunk * self.tokens_per_slot
            torch.manual_seed(42)
            scores = torch.randn(B, L_new, device=device)
            k = min(total_to_keep, L_new)
            _, topk_idx = torch.topk(scores, k=k, dim=1)
            idx_exp = topk_idx.unsqueeze(-1).expand(B, k, H * D)
            new_slot_k = torch.gather(features_k, 1, idx_exp).view(B, k, H, D)
            new_slot_v = torch.gather(features_v, 1, idx_exp).view(B, k, H, D)
            return {
                "new_slot_k": new_slot_k,
                "new_slot_v": new_slot_v,
                "slots": None,
                "masks": None,
            }

        feat_adapted = self.slot_input_adapter(features_k)

        prev_clean = self.prev_clean_slots.get(layer_idx, None)
        prev_hit = 1.0 if prev_clean is not None else 0.0
        if prev_clean is not None:
            slots_init = prev_clean.detach().clone()
        else:
            slots_init = self.slot_initializer(B).to(
                device=device, dtype=feat_adapted.dtype
            )

        out = self.slot_grouper(slots_init, feat_adapted)
        slots = out["slots"]        # [B, S, D]
        masks = out["masks"]        # [B, S, L]

        # ---- slot_output_adapter 输出 slot prototype，用于 reconstruction cos_sim ----
        recon_from_slot = self.slot_output_adapter(slots)   # [B, S, H*D]

        feat_norm = F.normalize(features_k.detach(), p=2, dim=-1)
        recon_norm = F.normalize(recon_from_slot, p=2, dim=-1)

        cos_sim = torch.einsum("bld,bsd->bsl", feat_norm, recon_norm).clamp(-1.0, 1.0)
        recon_err = 1.0 - cos_sim

        # ---- Soft selection score ----
        selection_score = (
            1.5 * cos_sim
            + self.selection_mask_log_weight
            * torch.log(masks.clamp_min(self.selection_eps))
        )

        # ---- STE hard mask（当前仅用于诊断/未来 recon 用） ----
        top1_slot_idx = masks.argmax(dim=1)
        hard_masks = torch.zeros_like(masks).scatter_(
            1, top1_slot_idx.unsqueeze(1), 1.0
        )
        ste_masks = hard_masks.detach() - masks.detach() + masks

        top1_conf = masks.max(dim=1).values
        conf_threshold = 0.5
        recon_valid = (top1_conf > conf_threshold).float()
        recon_loss_mask = top1_conf.detach().clamp_min(0.1)

        # ---- Token 选择 ----
        total_keep = self.num_slots_per_chunk * self.tokens_per_slot
        best_indices_1d, best_indices = self.select_tokens_by_reconstruction(
            features_k=features_k,
            slots=slots,
            masks=masks,
            tokens_per_slot=self.tokens_per_slot,
            selection_score=selection_score,
        )
        best_indices_expanded = best_indices_1d.unsqueeze(-1).expand(
            B, total_keep, H * D
        )

        # ---- 诊断日志 ----
        if self.enable_diagnostics:
            diag = self.compute_slot_diagnostics(
                masks=masks.detach(),
                best_indices=best_indices.detach(),
                recon_valid=recon_valid.detach(),
                layer_idx=layer_idx,
            )
            self.last_diagnostics = diag
            self.accumulated_diag["slot_entropy"]     += diag["slot_entropy"]
            self.accumulated_diag["slot_top1_conf"]   += diag["slot_top1_conf"]
            self.accumulated_diag["slot_util_min"]    += diag["slot_util_min"]
            self.accumulated_diag["slot_util_max"]    += diag["slot_util_max"]
            self.accumulated_diag["slot_util_std"]    += diag["slot_util_std"]
            self.accumulated_diag["slot_overlap"]     += diag["slot_overlap"]
            self.accumulated_diag["valid_token_ratio"] += diag["valid_token_ratio"]
            self.diag_count += 1
            self.diagnostic_step += 1
            if self.diagnostic_step % self.diagnostic_interval == 0:
                print(f"[SlotDiag][Layer {layer_idx}] entropy={diag['slot_entropy']:.4f}, ...")

        # ---- 提取被选中的 K/V tokens ----
        new_slot_k = torch.gather(features_k, 1, best_indices_expanded).view(
            B, self.num_slots_per_chunk * self.tokens_per_slot, H, D
        )
        new_slot_v = torch.gather(features_v, 1, best_indices_expanded).view(
            B, self.num_slots_per_chunk * self.tokens_per_slot, H, D
        )

        # ============================================================
        # 状态机更新 & 训练损失
        # ============================================================
        con_loss_val      = 0.0
        distill_loss_val  = 0.0
        recon_loss_val    = 0.0
        stage2_flag       = float(self.enable_dynamics_loss)

        if not slots.requires_grad:
            # ---------- rollout：只更新双缓冲状态机，不算 loss ----------
            if layer_idx in self.current_clean_slots:
                self.prev_clean_slots[layer_idx] = self.current_clean_slots[layer_idx]
            self.current_clean_slots[layer_idx] = slots.detach().clone()
        else:
            # ---------- training ----------
            loss_terms = []

            # ---- 1. temporal slot contrastive loss ----
            if prev_clean is not None:
                con_loss = self.slot_loss_fn(prev_clean.detach(), slots)
                loss_terms.append(self.con_loss_weight * con_loss)
                con_loss_val = con_loss.detach().item()

            # ---- 2. output-adapter reconstruction loss（来自方法一） ----
            if self.enable_dynamics_loss:
                # decoded_token_feat: [B, L, H*D]
                decoded_token_feat = torch.einsum(
                    "bsl,bsd->bld",
                    ste_masks,
                    recon_from_slot,
                )

                decoded_norm = F.normalize(decoded_token_feat, p=2, dim=-1)
                target_norm = F.normalize(features_k.detach(), p=2, dim=-1)

                decoder_cos_loss_matrix = 1.0 - (
                    decoded_norm * target_norm
                ).sum(dim=-1)  # [B, L_new]

                # ========== 面积加权（AW-MSE 风格） ==========
                area_k = hard_masks.sum(dim=-1)                    # [B, S]
                area_ratio = area_k / float(L_new)                 # [B, S]

                w = area_ratio.clamp_min(self.aw_eps).pow(
                    -self.aw_power_p
                )  # [B, S]

                if self.aw_norm_mode == "mean1":
                    w = w / w.mean(dim=1, keepdim=True).clamp_min(self.aw_eps)
                elif self.aw_norm_mode == "sum1":
                    w = self.num_slots_per_chunk * w / w.sum(
                        dim=1,
                        keepdim=True,
                    ).clamp_min(self.aw_eps)

                if self.aw_cap_w is not None:
                    w = w.clamp(max=self.aw_cap_w)

                # 每个 token 使用其 hard-assigned slot 的面积权重
                w_token = torch.gather(
                    w,
                    1,
                    top1_slot_idx,
                )  # [B, L_new]

                weighted_loss = decoder_cos_loss_matrix * recon_loss_mask * w_token
                weighted_denom = (recon_loss_mask * w_token).sum() + self.aw_eps

                decoder_recon_loss = weighted_loss.sum() / weighted_denom

                loss_terms.append(
                    self.recon_loss_weight * decoder_recon_loss
                )
                recon_loss_val = decoder_recon_loss.detach().item()
            
            # ---- 3. self-query selection distillation ----
            if self.enable_dynamics_loss and query is not None:
                # teacher 完全在 no_grad 下计算，避免构造 [B,H,Lq,L,D] 大张量
                with torch.no_grad():
                    q = query.permute(0, 2, 1, 3)              # [B,H,Lq,D]
                    k = new_k.detach().permute(0, 2, 1, 3)     # [B,H,L,D]
                    v = new_v.detach().permute(0, 2, 1, 3)     # [B,H,L,D]

                    scale = D ** -0.5
                    attn_probs = F.softmax(
                        torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1
                    )                                           # [B,H,Lq,L]

                    v_norm = v.norm(dim=-1)                     # [B,H,L]
                    token_contrib = torch.einsum(
                        'bhql,bhl->bl', attn_probs, v_norm
                    )
                    token_contrib = token_contrib / (H * attn_probs.shape[2])
                    I = token_contrib / (
                        token_contrib.sum(dim=-1, keepdim=True) + 1e-8
                    )                                           # [B,L]

                    T = masks.detach() * I.unsqueeze(1)         # [B,S,L]
                    T = T / (T.sum(dim=-1, keepdim=True) + 1e-8)

                P = F.softmax(selection_score / self.selection_temperature, dim=-1)
                loss_select = -(T * torch.log(P + 1e-8)).sum(dim=-1).mean()

                loss_terms.append(self.selection_distill_weight * loss_select)
                distill_loss_val = loss_select.detach().item()

            # ---- 3. 统一累加（loss_count 只在此处 +1） ----
            if len(loss_terms) > 0:
                loss = sum(loss_terms)
                if not isinstance(self.accumulated_loss, torch.Tensor):
                    self.accumulated_loss = loss
                else:
                    self.accumulated_loss = self.accumulated_loss + loss

                self.loss_count += 1
                self.accumulated_con_loss    += con_loss_val
                self.accumulated_recon_loss  += recon_loss_val
                self.accumulated_prev_hit    += prev_hit

            # distill 单独统计（避免被 con-only 的 forward 稀释）
            if distill_loss_val > 0.0:
                self.accumulated_dyn_loss += distill_loss_val
                self.distill_count += 1

        # ---- 4. 统一写一次 last_loss_breakdown，不再被 con 分支覆盖 ----
        self.last_loss_breakdown = {
            "slot_con":      con_loss_val,
            "slot_recon":    recon_loss_val,
            "slot_dyn":      distill_loss_val,
            "slot_prev_hit": prev_hit,
            "slot_stage2":   stage2_flag,
        }

        # ============================================================
        # 写入分数 write_score
        # ============================================================
        slot_coverage = masks.sum(dim=-1)                        # [B,S]
        slot_mass_ratio = slot_coverage / (
            slot_coverage.sum(dim=-1, keepdim=True) + 1e-8
        )

        mask_weight = masks / (masks.sum(dim=-1, keepdim=True) + 1e-6)
        avg_recon_sim = (cos_sim.detach() * mask_weight).sum(dim=-1)  # [B,S]
        recon_quality = ((avg_recon_sim + 1.0) * 0.5).clamp(0.0, 1.0)

        write_score = (slot_mass_ratio + 0.5 * recon_quality).clamp(0.0, 1.0)

        return {
            "new_slot_k": new_slot_k,
            "new_slot_v": new_slot_v,
            "slots": slots,
            "masks": masks,
            "write_score": write_score.detach(),
        }

    def get_and_reset_all_loss_stats(self):
        device = next(self.parameters()).device
        is_stage2 = float(self.enable_dynamics_loss)

        if self.loss_count == 0:
            return {
                "slot_total_tensor": torch.tensor(0.0, device=device),
                "slot_con": 0.0,
                "slot_recon": 0.0,
                "slot_dyn": 0.0,
                "slot_prev_hit": 0.0,
                "slot_stage2": is_stage2,
            }

        avg_total = self.accumulated_loss / self.loss_count
        out = {
            "slot_total_tensor": avg_total,
            "slot_con": self.accumulated_con_loss / self.loss_count,
            "slot_recon": self.accumulated_recon_loss / self.loss_count,
            "slot_dyn":   self.accumulated_dyn_loss   / max(self.distill_count, 1),  # ← 改
            "slot_prev_hit": self.accumulated_prev_hit / self.loss_count,
            "slot_stage2": is_stage2,
        }

        self.accumulated_loss = 0.0
        self.loss_count = 0
        self.accumulated_con_loss = 0.0
        self.accumulated_recon_loss = 0.0
        self.accumulated_dyn_loss = 0.0
        self.accumulated_prev_hit = 0.0
        self.distill_count = 0
        self.reset_diagnostics_only()

        return out
        
    def reset_sequence_state(self):
        """新视频开始时清空记忆"""
        self.prev_clean_slots.clear()
        self.current_clean_slots.clear()
        self.slot_write_scores.clear()
        # self.accumulated_loss = 0.0
        # self.loss_count = 0

        # self.accumulated_con_loss = 0.0
        # self.accumulated_recon_loss = 0.0
        # self.accumulated_dyn_loss = 0.0
        # self.accumulated_prev_hit = 0.0
        
        # self.reset_diagnostics_only()
    
    def reset_diagnostics_only(self):
        self.accumulated_diag = {
            "slot_entropy": 0.0,
            "slot_top1_conf": 0.0,
            "slot_util_min": 0.0,
            "slot_util_max": 0.0,
            "slot_util_std": 0.0,
            "slot_overlap": 0.0,
            "valid_token_ratio": 0.0,
        }
        self.diag_count = 0