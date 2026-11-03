# model/entity_bank.py
"""
V1 Persistent EntityBank
========================
- Entity Key 回答 WHO；Raw K/V Payload 回答 WHAT。
- 只做 select / store / retrieve / concatenate，不重投影 K/V。
- 每层独立 bank，生命周期: EMPTY -> TENTATIVE -> ACTIVE -> DORMANT。
- Read 是纯读；Write 以 chunk 为单位，每 chunk 一次。
- V1 假设推理 batch size = 1（长视频逐帧生成场景）。
"""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class PersistentEntityBank:
    EMPTY = 0
    TENTATIVE = 1
    ACTIVE = 2
    DORMANT = 3

    def __init__(
        self,
        num_slots_per_chunk: int = 7,
        tokens_per_slot: int = 64,
        slot_dim: int = 64,
        E_max: int = 32,
        M_hist: int = 4,
        # ---------- matching ----------
        tau_active: float = 0.60,
        tau_dormant: float = 0.75,
        lambda_idx: float = 0.03,          # 极弱的 slot-index prior
        # ---------- lifecycle ----------
        confirm_hits: int = 2,
        tentative_max_miss: int = 1,
        active_to_dormant_miss: int = 2,
        # ---------- identity EMA ----------
        eta_base: float = 0.10,
        dormant_update_scale: float = 0.25,
        identity_update_threshold: float = 0.55,
        # ---------- payload writing ----------
        write_threshold: float = 0.30,
        write_match_threshold: float = 0.50,
        # ---------- retrieval quotas ----------
        matched_quota: int = 6,
        persistent_quota: int = 4,
        recent_quota: int = 4,
        max_groups_per_entity: int = 2,
        # ---------- ablation switches ----------
        enabled: bool = True,              # 总开关
        use_retrieval: bool = True,        # A3/A4: 是否真正影响 attention
        write_payload: bool = True,        # A2: 是否写 raw KV payload
        allow_dormant_reid: bool = False,
    ):
        self.num_slots_per_chunk = num_slots_per_chunk
        self.tokens_per_slot = tokens_per_slot
        self.slot_dim = slot_dim
        self.E_max = E_max
        self.M_hist = M_hist

        self.tau_active = tau_active
        self.tau_dormant = tau_dormant
        self.lambda_idx = lambda_idx

        self.confirm_hits = confirm_hits
        self.tentative_max_miss = tentative_max_miss
        self.active_to_dormant_miss = active_to_dormant_miss

        self.eta_base = eta_base
        self.dormant_update_scale = dormant_update_scale
        self.identity_update_threshold = identity_update_threshold

        self.write_threshold = write_threshold
        self.write_match_threshold = write_match_threshold

        self.matched_quota = matched_quota
        self.persistent_quota = persistent_quota
        self.recent_quota = recent_quota
        self.max_groups_per_entity = max_groups_per_entity

        self.enabled = enabled
        self.use_retrieval = use_retrieval
        self.write_payload = write_payload
        self.allow_dormant_reid = allow_dormant_reid

        # 每层独立 bank，lazy init（device / H / D 由第一次 commit 确定）
        self.banks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._batch_size = 1

        # debug 统计
        self.last_stats = {}
        
        # ==================== A1 阈值标定诊断 ====================
        self.enable_diagnostics = True
        self.concise_log_interval = 10      # 每 10 个 chunk 打一行精简日志
        self.reset_diagnostics()
        
    def reset_diagnostics(self):
        self.diag = {
            # ---- 分布（用伪标签 Hungarian，避免 tau 截断） ----
            "same_cos": [],                # 伪标签 same-ID
            "cross_cos": [],               # 伪标签 cross-ID
            "active_cos": [],              # same-ID 且 prev==ACTIVE
            "tentative_cos": [],           # same-ID 且 prev==TENTATIVE
            "dormant_cos": [],             # same-ID 且 prev==DORMANT (re-ID)
            "first_match_cos": [],         # same-ID 且 pre-hits==0
            "stable_match_cos": [],        # same-ID 且 pre-hits>=2
            "unmatched_max_cos": [],       # 实际未匹配 slot 的最大相似度（误拒风险）
            # ---- 累计计数 ----
            "num_matches": 0,
            "num_reid": 0,
            "num_tentative_births": 0,
            "num_births": 0,
            "num_evictions": 0,
            "num_unmatched_slots": 0,
            # ---- state 分布（按 chunk 累加） ----
            "num_active": 0,
            "num_dormant": 0,
            "num_tentative": 0,
            "num_entities": 0,
            "num_payload_writes": 0,        # 新增：写入 snapshot 的次数
            "dormant_probe_cos": [],        # 新增：DORMANT 探测余弦（不参与匹配）
        }
        self.diag_step = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_sequence(self, batch_size: int = 1, print_summary: bool = False):
        """新视频开始时调用，清空所有层 bank。与 prompt 无关。"""
        if self.enable_diagnostics and self.diag_step > 0:
            if print_summary:
                self.print_diagnostic_summary(reset=True)   # ← 自动打印上个序列的汇总
            else:
                self.reset_diagnostics()
        else:
            self.reset_diagnostics()
        self.banks.clear()
        self._batch_size = batch_size
        self.last_stats = {}

    def _ensure_bank(self, layer_idx: int, B: int, device, dtype, H: int, D: int):
        bank = self.banks.get(layer_idx)
        if bank is None:
            E, M, R = self.E_max, self.M_hist, self.tokens_per_slot
            bank = {
                "entity_key": torch.zeros(B, E, self.slot_dim, device=device, dtype=dtype),
                "entity_state": torch.zeros(B, E, dtype=torch.long, device=device),
                "entity_hits": torch.zeros(B, E, dtype=torch.long, device=device),
                "entity_miss": torch.zeros(B, E, dtype=torch.long, device=device),
                "entity_quality": torch.zeros(B, E, device=device, dtype=dtype),
                "entity_last_slot": torch.full((B, E), -1, dtype=torch.long, device=device),
                "entity_last_seen": torch.full((B, E), -1, dtype=torch.long, device=device),
                "entity_birth_hits": torch.zeros(B, E, dtype=torch.long, device=device),
                "entity_dormant_age": torch.zeros(B, E, device=device, dtype=dtype),
                "entity_k_hist": torch.zeros(B, E, M, R, H, D, device=device, dtype=dtype),
                "entity_v_hist": torch.zeros(B, E, M, R, H, D, device=device, dtype=dtype),
                "entity_hist_valid": torch.zeros(B, E, M, dtype=torch.bool, device=device),
                "entity_hist_quality": torch.zeros(B, E, M, device=device, dtype=dtype),
                "entity_hist_step": torch.full((B, E, M), -1, dtype=torch.long, device=device),
                "entity_hist_ptr": torch.zeros(B, E, dtype=torch.long, device=device),
            }
            self.banks[layer_idx] = bank
        return bank

    # ------------------------------------------------------------------
    # Commit —— 每个完整 chunk 调用一次，纯 no_grad
    # ------------------------------------------------------------------
    @torch.no_grad()
    def commit_chunk(
        self,
        layer_idx: int,
        current_slots: torch.Tensor,     # [B, S, slot_dim]
        slot_k: torch.Tensor,            # [B, S, R, H, D]
        slot_v: torch.Tensor,            # [B, S, R, H, D]
        write_score: torch.Tensor,       # [B, S]
        step: int,
    ):
        if not self.enabled:
            return
        if current_slots is None or slot_k is None or write_score is None:
            return

        B, S, R, H, D = slot_k.shape
        device = slot_k.device
        dtype = slot_k.dtype
        bank = self._ensure_bank(layer_idx, B, device, dtype, H, D)

        current_key = F.normalize(current_slots.detach(), dim=-1)          # [B,S,dim]

        # 1. similarity ---------------------------------------------------
        sim = torch.einsum("bsd,bed->bse", current_key, bank["entity_key"])  # [B,S,E]
        
        # ===== A1: 保存未加 slot-index prior 的原始相似度，供无偏阈值标定 =====
        raw_sim = sim.clone()

        # 2. 极弱 slot-index prior（只给 ACTIVE 且上一步刚出现的实体）--------
        recent_act = (
            (bank["entity_state"] == self.ACTIVE)
            & (bank["entity_last_seen"] == step - 1)
            & (bank["entity_quality"] > 0)
        )  # [B,E]
        if recent_act.any():
            for b in range(B):
                for e in recent_act[b].nonzero(as_tuple=False).flatten().tolist():
                    sp = bank["entity_last_slot"][b, e].item()
                    if 0 <= sp < S:
                        sim[b, sp, e] = sim[b, sp, e] + self.lambda_idx

        # 3. state-dependent gating ---------------------------------------
        tau_per_entity = torch.where(
            bank["entity_state"] == self.DORMANT,
            torch.tensor(self.tau_dormant, device=device),
            torch.tensor(self.tau_active, device=device),
        )  # [B,E]
        valid_mask = (bank["entity_state"] != self.EMPTY).unsqueeze(1)      # [B,1,E]
        if not self.allow_dormant_reid:
            valid_mask = valid_mask & (bank["entity_state"] != self.DORMANT).unsqueeze(1)

        # 4. gated Hungarian + unmatched -----------------------------------
        for b in range(B):
            matches, unmatched_slots, unmatched_entities = self._gated_hungarian(
                sim[b], valid_mask[b], tau_per_entity[b]
            )

            # ===== A1: 采集诊断 =====
            if self.enable_diagnostics:
                diag_tau = torch.full_like(tau_per_entity[b], -1e9)
                diag_matches, _, _ = self._gated_hungarian(
                    sim[b], valid_mask[b], diag_tau, dummy_value=-10.0
                )
                self._collect_match_diagnostics(
                    b, matches, diag_matches, unmatched_slots, bank, raw_sim[b]
                )

            self._update_matched(b, matches, bank, current_key, sim[b],
                                 slot_k, slot_v, write_score, step)
            self._update_missing(b, unmatched_entities, bank, step)
            matched_entities = {e for _, e, _ in matches}

            self._update_tentative(
                b=b,
                unmatched_slots=unmatched_slots,
                bank=bank,
                current_key=current_key,
                sim_b=sim[b],
                slot_k=slot_k,
                slot_v=slot_v,
                write_score=write_score,
                step=step,
                matched_entities=matched_entities,
            )

        # 5. birth promotion / eviction ------------------------------------
        self._promote_tentatives(bank)
        for b in range(B):
            self._evict_if_needed(b, bank)

        # 6. debug stats ---------------------------------------------------
        st = bank["entity_state"]
        self.last_stats = {
            "num_active": int((st == self.ACTIVE).sum()),
            "num_dormant": int((st == self.DORMANT).sum()),
            "num_tentative": int((st == self.TENTATIVE).sum()),
            "num_entities": int((st != self.EMPTY).sum()),
            "hist_fill_rate": float(bank["entity_hist_valid"].float().mean().item()),
        }
        
        # ===== A1: 进度日志 =====
        if self.enable_diagnostics:
            self.diag_step += 1
            if self.concise_log_interval > 0 and self.diag_step % self.concise_log_interval == 0:
                self.log_concise_stats()
    
    @torch.no_grad()
    def _collect_match_diagnostics(
        self,
        b: int,
        matches: List[Tuple[int, int, float]],
        diag_matches: List[Tuple[int, int, float]],
        unmatched_slots: List[int],
        bank: Dict[str, torch.Tensor],
        raw_sim_b: torch.Tensor,          # [S,E] 无 prior 的原始 cosine
    ):
        state = bank["entity_state"][b]        # [E]
        valid = (state != self.EMPTY)

        # 1) 实际（受当前阈值影响）匹配计数
        for s, e, _conf in matches:
            self.diag["num_matches"] += 1
            if state[e].item() == self.DORMANT:
                self.diag["num_reid"] += 1

        # 2) 无阈值伪标签 Hungarian -> 无偏 same/cross 分布
        for s, e, _conf in diag_matches:
            conf = float(raw_sim_b[s, e].item())     # 用原始 cosine，去掉 prior

            prev_state = state[e].item()
            self.diag["same_cos"].append(conf)
            if prev_state == self.ACTIVE:
                self.diag["active_cos"].append(conf)
            elif prev_state == self.DORMANT:
                self.diag["dormant_cos"].append(conf)
            elif prev_state == self.TENTATIVE:
                self.diag["tentative_cos"].append(conf)

            pre_hits = int(bank["entity_hits"][b, e].item())
            if pre_hits == 0:
                self.diag["first_match_cos"].append(conf)
            elif pre_hits >= 2:
                self.diag["stable_match_cos"].append(conf)

            # 该 slot 与所有其它 valid entity 的相似度 -> cross-ID
            other = valid.clone()
            other[e] = False
            if other.any():
                self.diag["cross_cos"].extend(raw_sim_b[s][other].tolist())

        # 3) 实际未匹配 slot -> 最大相似度（潜在误拒 / 新实体风险）
        for s in unmatched_slots:
            self.diag["num_unmatched_slots"] += 1
            if valid.any():
                self.diag["unmatched_max_cos"].append(
                    float(raw_sim_b[s][valid].max().item())
                )

        # 4) DORMANT 探测：不参与匹配，只记录余弦分布（供 V2 re-ID 标定）
        if not self.allow_dormant_reid:
            dor_idx = (state == self.DORMANT).nonzero(as_tuple=False).flatten()
            if len(dor_idx) > 0:
                for s in unmatched_slots:
                    self.diag["dormant_probe_cos"].extend(
                        raw_sim_b[s][dor_idx].tolist()
                    )

        # 4) state 分布
        self.diag["num_active"] += int((state == self.ACTIVE).sum().item())
        self.diag["num_dormant"] += int((state == self.DORMANT).sum().item())
        self.diag["num_tentative"] += int((state == self.TENTATIVE).sum().item())
        self.diag["num_entities"] += int(valid.sum().item())

    # ------------------------------------------------------------------
    # Retrieve —— 纯读，不改任何 bank 状态
    # ------------------------------------------------------------------
    @torch.no_grad()
    def retrieve(
        self,
        layer_idx: int,
        current_slots: torch.Tensor,     # [B, S, slot_dim]
        max_groups: int,
        step: int = 0,
    ) -> Dict:
        if not (self.enabled and self.use_retrieval):
            return self._empty_retrieval()
        bank = self.banks.get(layer_idx)
        if bank is None or current_slots is None or max_groups <= 0:
            return self._empty_retrieval()

        B, S, _ = current_slots.shape
        if B != 1:
            # V1 简化：推理建议 B=1；多 batch 需按 batch 分别 gather 并 pad
            print("[WARN] EntityBank V1 retrieve assumes B=1; using batch 0.")
        b0 = 0
        device = current_slots.device
        state = bank["entity_state"][b0]
        valid = state != self.EMPTY
        if not valid.any():
            return self._empty_retrieval()

        current_key = F.normalize(current_slots[b0].detach(), dim=-1)      # [S,dim]
        ent_key = bank["entity_key"][b0]                                    # [E,dim]
        sim = torch.einsum("sd,ed->se", current_key, ent_key)               # [S,E]

        sim_masked = sim.masked_fill(~valid.unsqueeze(0), -1e9)
        visual_rel = sim_masked.max(dim=0).values                           # [E]

        # 当前 chunk 直接命中的实体集合（matched）
        valid_for_current_match = valid.clone()

        # V1: DORMANT 不参与 current matched 判断
        # DORMANT 只能通过 persistent quota 被读出
        if not self.allow_dormant_reid:
            valid_for_current_match = valid_for_current_match & (state != self.DORMANT)

        if valid_for_current_match.any():
            sim_for_match = sim.masked_fill(
                ~valid_for_current_match.unsqueeze(0),
                -1e9,
            )

            best_e = sim_for_match.argmax(dim=-1)
            best_sim = sim_for_match.max(dim=-1).values

            matched_entities = set(
                best_e[best_sim > self.tau_active].tolist()
            )
        else:
            matched_entities = set()

        # entity-level score
        score_e = visual_rel.clone()
        score_e = score_e + 0.10 * bank["entity_quality"][b0]
        score_e = score_e + 0.05 * torch.log1p(bank["entity_hits"][b0].float())
        for e in matched_entities:
            score_e[e] = score_e[e] + 0.15
        score_e = score_e.masked_fill(~valid, -1e9)

        # ------------------------------------------------------------
        # 候选 snapshots
        # c = {
        #   "score": snapshot_score,
        #   "entity": e,
        #   "hist": m,
        #   "quality": q,
        #   "step": st,
        #   "state": state_e,
        #   "matched": bool,
        # }
        # ------------------------------------------------------------
        cands = []

        for e in range(self.E_max):
            if not valid[e]:
                continue

            vm = bank["entity_hist_valid"][b0, e]
            if not vm.any():
                continue

            entity_score = score_e[e].item()
            state_e = int(state[e].item())
            is_matched = e in matched_entities

            for m in vm.nonzero(as_tuple=False).flatten().tolist():
                q = float(bank["entity_hist_quality"][b0, e, m].item())
                stp = int(bank["entity_hist_step"][b0, e, m].item())

                age = max(step - stp, 0)

                # recent bonus：越新越高，最多 0.05
                recent_bonus = 0.05 * max(0.0, 1.0 - min(age, 50) / 50.0)

                snap_score = (
                    entity_score
                    + 0.20 * q
                    + recent_bonus
                    + (0.15 if is_matched else 0.0)
                )

                cands.append({
                    "score": snap_score,
                    "entity": e,
                    "hist": m,
                    "quality": q,
                    "step": stp,
                    "state": state_e,
                    "matched": is_matched,
                })

        cands.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        selected_keys = set()
        per_entity_count: Dict[int, int] = {}

        def can_add(c):
            key = (c["entity"], c["hist"])
            if key in selected_keys:
                return False
            if per_entity_count.get(c["entity"], 0) >= self.max_groups_per_entity:
                return False
            if len(selected) >= max_groups:
                return False
            return True

        def add_candidate(c):
            if not can_add(c):
                return False
            key = (c["entity"], c["hist"])
            selected.append(c)
            selected_keys.add(key)
            per_entity_count[c["entity"]] = per_entity_count.get(c["entity"], 0) + 1
            return True

        # ---------------- round 1: matched quota ----------------
        n_added = 0
        for c in cands:
            if n_added >= self.matched_quota:
                break
            if c["matched"]:
                if add_candidate(c):
                    n_added += 1

        # ---------------- round 2: persistent quota ----------------
        # ACTIVE / DORMANT 都可以被 persistent quota 读取
        # 但 DORMANT 不参与 re-ID matching
        n_added = 0
        for c in cands:
            if n_added >= self.persistent_quota:
                break
            if c["matched"]:
                continue
            if c["state"] in (self.ACTIVE, self.DORMANT):
                if add_candidate(c):
                    n_added += 1

        # ---------------- round 3: recent-quality quota ----------------
        recent_cands = sorted(
            cands,
            key=lambda x: (x["quality"], x["step"]),
            reverse=True,
        )

        n_added = 0
        for c in recent_cands:
            if n_added >= self.recent_quota:
                break
            if add_candidate(c):
                n_added += 1

        # ---------------- global backfill ----------------
        for c in cands:
            if len(selected) >= max_groups:
                break
            add_candidate(c)

        if not selected:
            return self._empty_retrieval()

        # gather raw K/V
        k_list, v_list, q_list = [], [], []
        entity_ids, snapshot_ids = [], []

        for c in selected:
            e = c["entity"]
            m = c["hist"]
            q = c["quality"]

            k_snap = bank["entity_k_hist"][b0, e, m]
            v_snap = bank["entity_v_hist"][b0, e, m]

            R, H, D = k_snap.shape

            k_list.append(k_snap)
            v_list.append(v_snap)
            q_list.append(q)
            entity_ids.append(e)
            snapshot_ids.append(m)

        ret_k = torch.stack(k_list, dim=0).unsqueeze(0).view(
            1, len(selected) * R, H, D
        )
        ret_v = torch.stack(v_list, dim=0).unsqueeze(0).view(
            1, len(selected) * R, H, D
        )

        quals = torch.tensor(
            q_list,
            device=device,
            dtype=ret_k.dtype,
        ).view(1, -1)

        return {
            "k": ret_k,
            "v": ret_v,
            "num_groups": len(selected),
            "hist_qualities": quals,
            "entity_ids": entity_ids,
            "snapshot_ids": snapshot_ids,
        }

    def _empty_retrieval(self):
        return {
            "k": None,
            "v": None,
            "num_groups": 0,
            "hist_qualities": None,
            "entity_ids": [],
            "snapshot_ids": [],
        }

    # ------------------------------------------------------------------
    # 内部 helpers
    # ------------------------------------------------------------------
    def _gated_hungarian(
        self,
        sim: torch.Tensor,             # [S,E]
        valid_mask: torch.Tensor,      # [1,E] bool
        tau_per_entity: torch.Tensor,  # [E]
        dummy_value: Optional[float] = None,    # 新增：诊断用无阈值 Hungarian
    ):
        S, E = sim.shape
        gate = sim >= tau_per_entity.unsqueeze(0)              # [S,E]
        cost = sim.masked_fill(~(valid_mask & gate), -1e9)
        # 拼 dummy 实体列：值 = tau_active，表示“这个 slot 不匹配任何实体”
        if dummy_value is None:
            dummy_value = self.tau_active
        cost_aug = torch.cat(
            [cost, cost.new_full((S, 1), dummy_value)], dim=1
        )

        if _HAS_SCIPY:
            cost_np = cost_aug.float().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np, maximize=True)
        else:
            row_ind, col_ind = self._greedy_match(cost_aug.cpu().numpy())

        matches, unmatched_slots, unmatched_entities = [], list(range(S)), list(range(E))
        for r, c in zip(row_ind, col_ind):
            if c < E and cost[r, c] > tau_per_entity[c].item():
                matches.append((int(r), int(c), float(sim[r, c])))
                if r in unmatched_slots:
                    unmatched_slots.remove(r)
                if c in unmatched_entities:
                    unmatched_entities.remove(c)
        return matches, unmatched_slots, unmatched_entities

    @staticmethod
    def _greedy_match(cost_np):
        import numpy as np
        S, C = cost_np.shape
        rows, cols = list(range(S)), list(range(C))
        pairs = []
        while rows and cols:
            sub = cost_np[np.ix_(rows, cols)]
            i, j = np.unravel_index(np.argmax(sub), sub.shape)
            r, c = rows[i], cols[j]
            pairs.append((r, c))
            rows.remove(r)
            cols.remove(c)
        return pairs

    # ------------------------- matched -------------------------
    def _update_matched(self, b, matches, bank, current_key, sim_b,
                        slot_k, slot_v, write_score, step):
        for s, e, conf in matches:
            prev_state = bank["entity_state"][b, e].item()
            q = float(write_score[b, s].clamp(0, 1).detach())
            alpha = self.eta_base * q * float(min(max(conf, 0.0), 1.0))
            if prev_state == self.DORMANT:
                alpha *= self.dormant_update_scale

            bank["entity_miss"][b, e] = 0
            bank["entity_hits"][b, e] += 1
            bank["entity_last_slot"][b, e] = s
            bank["entity_last_seen"][b, e] = step
            bank["entity_dormant_age"][b, e] = 0
            bank["entity_quality"][b, e] = 0.9 * bank["entity_quality"][b, e] + 0.1 * q

            if prev_state == self.TENTATIVE:
                bank["entity_birth_hits"][b, e] += 1
                if bank["entity_birth_hits"][b, e] >= self.confirm_hits:
                    bank["entity_state"][b, e] = self.ACTIVE
                    if self.enable_diagnostics:
                        self.diag["num_births"] += 1
            elif prev_state != self.EMPTY:
                bank["entity_state"][b, e] = self.ACTIVE

            # identity EMA（re-ID 后第一次不更新，更保守）
            if prev_state == self.DORMANT:
                pass
            elif conf >= self.identity_update_threshold:
                old = bank["entity_key"][b, e]
                new = F.normalize((1.0 - alpha) * old + alpha * current_key[b, s], dim=-1)
                bank["entity_key"][b, e] = new

            # payload write（match / identity update / payload 三个决策分离）
            if (self.write_payload
                    and q >= self.write_threshold
                    and conf >= self.write_match_threshold):
                self._write_snapshot(b, e, bank, slot_k[b, s], slot_v[b, s], q, step)

    # ------------------------- missing -------------------------
    def _update_missing(self, b, unmatched_entities, bank, step):
        for e in unmatched_entities:
            st = bank["entity_state"][b, e].item()
            if st == self.EMPTY:
                continue
            if st == self.ACTIVE:
                bank["entity_miss"][b, e] += 1
                if bank["entity_miss"][b, e] >= self.active_to_dormant_miss:
                    bank["entity_state"][b, e] = self.DORMANT
                    bank["entity_miss"][b, e] = 0
            elif st == self.DORMANT:
                bank["entity_dormant_age"][b, e] += 1
            elif st == self.TENTATIVE:
                bank["entity_miss"][b, e] += 1
                if bank["entity_miss"][b, e] > self.tentative_max_miss:
                    self._clear_entity(b, e, bank)

    # ------------------------- tentative birth -------------------------
    def _update_tentative(
        self,
        b,
        unmatched_slots,
        bank,
        current_key,
        sim_b,
        slot_k,
        slot_v,
        write_score,
        step,
        matched_entities=None,
    ):
        matched_entities = matched_entities or set()

        for s in unmatched_slots:
            q = float(write_score[b, s].clamp(0, 1).detach())

            if q < self.write_threshold:
                continue

            # duplicate suppression 只能阻挡明显低质量 fragment
            # 不能阻挡高质量相似新实体
            if len(matched_entities) > 0:
                matched_idx = torch.tensor(
                    list(matched_entities),
                    device=sim_b.device,
                    dtype=torch.long,
                )
                dup_sim = sim_b[s, matched_idx].max().item()

                # 只有非常像已有 matched entity 且自身质量不高时，才认为是 fragment
                if dup_sim > 0.97 and q < max(0.45, self.write_threshold * 1.5):
                    continue

            e_free = self._find_free_entity(bank, b)
            if e_free is None:
                if not self._evict_if_needed(b, bank):
                    continue
                e_free = self._find_free_entity(bank, b)
                if e_free is None:
                    continue
            bank["entity_key"][b, e_free] = F.normalize(current_key[b, s], dim=-1)
            bank["entity_state"][b, e_free] = self.TENTATIVE
            if self.enable_diagnostics:
                self.diag["num_tentative_births"] += 1
            bank["entity_birth_hits"][b, e_free] = 1
            bank["entity_quality"][b, e_free] = q
            bank["entity_hits"][b, e_free] = 0
            bank["entity_miss"][b, e_free] = 0
            bank["entity_last_slot"][b, e_free] = s
            bank["entity_last_seen"][b, e_free] = step
            bank["entity_dormant_age"][b, e_free] = 0
            if self.write_payload:
                self._write_snapshot(b, e_free, bank, slot_k[b, s], slot_v[b, s], q, step)

    def _promote_tentatives(self, bank):
        prom = (
            (bank["entity_state"] == self.TENTATIVE)
            & (bank["entity_birth_hits"] >= self.confirm_hits)
        )
        if self.enable_diagnostics:
            self.diag["num_births"] += int(prom.sum().item())
        bank["entity_state"][prom] = self.ACTIVE

    # ------------------------- capacity -------------------------
    def _find_free_entity(self, bank, b) -> Optional[int]:
        empty = (bank["entity_state"][b] == self.EMPTY).nonzero(as_tuple=False).flatten().tolist()
        return empty[0] if empty else None

    def _evict_if_needed(self, b, bank) -> bool:
        """只回收 DORMANT / 低价值 TENTATIVE；ACTIVE 永不驱逐。"""
        state = bank["entity_state"][b]
        if (state == self.EMPTY).any():
            return True
        candidates = []
        for e in range(self.E_max):
            st = state[e].item()
            if st == self.DORMANT:
                retain = (
                    0.5 * bank["entity_quality"][b, e]
                    + 0.1 * torch.log1p(bank["entity_hits"][b, e].float())
                    - 0.05 * bank["entity_dormant_age"][b, e]
                )
                candidates.append((retain.item(), e))
            elif st == self.TENTATIVE:
                q = bank["entity_quality"][b, e].item()
                if q < self.write_threshold * 0.9:
                    candidates.append((q, e))
        if not candidates:
            return False
        candidates.sort(key=lambda x: x[0])
        self._clear_entity(b, candidates[0][1], bank)
        if self.enable_diagnostics:
            self.diag["num_evictions"] += 1
        return True

    def _clear_entity(self, b, e, bank):
        bank["entity_state"][b, e] = self.EMPTY
        bank["entity_key"][b, e].zero_()
        bank["entity_hits"][b, e] = 0
        bank["entity_miss"][b, e] = 0
        bank["entity_quality"][b, e] = 0
        bank["entity_last_slot"][b, e] = -1
        bank["entity_last_seen"][b, e] = -1
        bank["entity_birth_hits"][b, e] = 0
        bank["entity_dormant_age"][b, e] = 0
        bank["entity_hist_valid"][b, e] = False
        bank["entity_hist_quality"][b, e] = 0
        bank["entity_hist_step"][b, e] = -1
        bank["entity_hist_ptr"][b, e] = 0

    # ------------------------- payload -------------------------
    def _write_snapshot(self, b, e, bank, k_grp, v_grp, q, step):
        valid = bank["entity_hist_valid"][b, e]
        if (~valid).any():
            pos = (~valid).nonzero(as_tuple=False)[0].item()
        else:
            # quality-aware replacement：低质量 + 较旧的 snapshot 优先被替换
            qual = bank["entity_hist_quality"][b, e]
            stp = bank["entity_hist_step"][b, e].float()
            replace_score = qual - 0.01 * ((step - stp).clamp(0) / 100.0)
            pos = replace_score.argmin().item()
        bank["entity_k_hist"][b, e, pos] = k_grp
        bank["entity_v_hist"][b, e, pos] = v_grp
        bank["entity_hist_valid"][b, e, pos] = True
        bank["entity_hist_quality"][b, e, pos] = q
        bank["entity_hist_step"][b, e, pos] = step
        if self.enable_diagnostics:
            self.diag["num_payload_writes"] = self.diag.get("num_payload_writes", 0) + 1
        bank["entity_hist_ptr"][b, e] = (pos + 1) % self.M_hist
    
    def log_concise_stats(self):
        if not self.diag["same_cos"]:
            print(f"[EntityBank-Diag] chunk={self.diag_step}  尚无匹配样本")
            return
        same_t = torch.tensor(self.diag["same_cos"][-100:], dtype=torch.float32)
        cross_t = None
        if self.diag["cross_cos"]:
            cross_t = torch.tensor(self.diag["cross_cos"][-200:], dtype=torch.float32)
        s = (
            f"[EntityBank-Diag] chunk={self.diag_step}  "
            f"matches={self.diag['num_matches']}  reid={self.diag['num_reid']}  "
            f"same.mean={same_t.mean().item():.4f}"
        )
        if cross_t is not None:
            s += f"  cross.mean={cross_t.mean().item():.4f}"
        s += (
            f"  active={self.last_stats.get('num_active', 0)}  "
            f"dormant={self.last_stats.get('num_dormant', 0)}  "
            f"tentative={self.last_stats.get('num_tentative', 0)}"
        )
        # print(s)
    
    @torch.no_grad()
    def suggest_thresholds(self):
        """从已收集分布给出 tau_active / tau_dormant 建议值。"""
        if len(self.diag["same_cos"]) == 0 or len(self.diag["cross_cos"]) == 0:
            return None, None

        same_t = torch.tensor(self.diag["same_cos"], dtype=torch.float32)
        cross_t = torch.tensor(self.diag["cross_cos"], dtype=torch.float32)
        same_p5 = torch.quantile(same_t, 0.05).item()
        cross_p95 = torch.quantile(cross_t, 0.95).item()

        # 平衡点：同ID的 p5 和 异ID的 p95 的中点
        tau_active = (same_p5 + cross_p95) / 2.0

        tau_dormant = None
        if len(self.diag["dormant_cos"]) > 0:
            dor_t = torch.tensor(self.diag["dormant_cos"], dtype=torch.float32)
            dor_p5 = torch.quantile(dor_t, 0.05).item()
            raw_dormant = (dor_p5 + cross_p95) / 2.0
            # 设计文档要求 DORMANT 更保守（更高），因此至少比 active 高 0.03
            tau_dormant = max(raw_dormant, tau_active + 0.03)

        return tau_active, tau_dormant

    def apply_suggested_thresholds(self):
        tau_a, tau_d = self.suggest_thresholds()
        if tau_a is not None:
            self.tau_active = tau_a
            print(f"[EntityBank-Diag] tau_active = {tau_a:.4f}")
        if tau_d is not None:
            self.tau_dormant = tau_d
            print(f"[EntityBank-Diag] tau_dormant = {tau_d:.4f}")
        return tau_a, tau_d
    
    @torch.no_grad()
    def print_diagnostic_summary(self, reset: bool = True):
        if self.diag_step == 0 or len(self.diag["same_cos"]) == 0:
            print(f"[EntityBank-Diag] 尚未收集匹配样本 (chunks={self.diag_step})，先多跑几个 chunk。")
            if reset:
                self.reset_diagnostics()
            return

        def stats(name, arr):
            if not arr:
                print(f"  {name}: <no samples>")
                return
            t = torch.tensor(arr, dtype=torch.float32)
            q = torch.quantile(t, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])).tolist()
            print(
                f"  {name}: n={t.numel():5d}  "
                f"min={t.min().item():.4f}  p5={q[0]:.4f}  p25={q[1]:.4f}  "
                f"med={q[2]:.4f}  p75={q[3]:.4f}  p95={q[4]:.4f}  max={t.max().item():.4f}  "
                f"mean={t.mean().item():.4f}"
            )

        n_chunk = max(self.diag_step, 1)
        total_slots = n_chunk * self.num_slots_per_chunk
        match_rate = self.diag["num_matches"] / max(total_slots, 1)

        print("=" * 74)
        print(f"  当前 tau_active={self.tau_active:.4f}  tau_dormant={self.tau_dormant:.4f}  "
              f"allow_dormant_reid={self.allow_dormant_reid}")
        print("[EntityBank-Diag] A1 阈值标定汇总")
        print(f"  chunk 数: {self.diag_step}   match 数: {self.diag['num_matches']} "
              f"(match_rate={match_rate:.3f})   re-ID 数: {self.diag['num_reid']}")
        print(f"  tentative 新建: {self.diag['num_tentative_births']}   "
              f"确认 birth: {self.diag['num_births']}   eviction: {self.diag['num_evictions']}")
        print(f"  payload 写入: {self.diag.get('num_payload_writes', 0)}   "
              f"hist 平均填充率: {self.last_stats.get('hist_fill_rate', 0):.3f}")
        print(f"  unmatched slot 数: {self.diag['num_unmatched_slots']}")
        print(f"  平均每 chunk: active={self.diag['num_active'] / n_chunk:.2f}  "
              f"dormant={self.diag['num_dormant'] / n_chunk:.2f}  "
              f"tentative={self.diag['num_tentative'] / n_chunk:.2f}  "
              f"total={self.diag['num_entities'] / n_chunk:.2f}")
        print()
        print("  === 匹配相似度 (same-ID) 分布 ===")
        stats("same-ID (伪标签)      ", self.diag["same_cos"])
        stats("  ├─ ACTIVE           ", self.diag["active_cos"])
        stats("  ├─ TENTATIVE        ", self.diag["tentative_cos"])
        stats("  └─ DORMANT (re-ID)  ", self.diag["dormant_cos"])
        stats("first-match (hits=0)  ", self.diag["first_match_cos"])
        stats("stable-match (hits>=2)", self.diag["stable_match_cos"])
        print()
        print("  === 非匹配相似度 (cross-ID) 分布 ===")
        stats("cross-ID (其它实体)   ", self.diag["cross_cos"])
        stats("unmatched max-sim     ", self.diag["unmatched_max_cos"])
        print()

        tau_a, tau_d = self.suggest_thresholds()
        if tau_a is not None:
            print(f"  [建议] tau_active  ≈ {tau_a:.4f}   (same p5 与 cross p95 的中点)")
        else:
            print("  [提示] 暂无 same/cross 样本，无法给建议")
        if tau_d is not None:
            print(f"  [建议] tau_dormant ≈ {tau_d:.4f}   (已 clamp 到 >= tau_active + 0.03)")
        else:
            print("  [提示] 暂无 DORMANT re-ID 样本；可先用 tau_dormant = tau_active + 0.10 起步")
        print("=" * 74)

        if reset:
            self.reset_diagnostics()