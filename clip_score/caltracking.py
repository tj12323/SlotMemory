import os
import torch
import torch.nn.functional as F
import einops
import numpy as np
from typing import Optional

# 从 metrics.py 导入核心函数（请将 metrics.py 放在同一目录下）
from slotcontrast.metrics import adjusted_rand_index, VideoIoU

import time
from contextlib import contextmanager
import multiprocessing as mp
from tqdm import tqdm

WORKER_GPU_ID = None

def init_worker(gpu_queue):
    """每个子进程启动时调用一次，从队列拿一个 GPU id 并绑定。"""
    global WORKER_GPU_ID
    WORKER_GPU_ID = gpu_queue.get()
    import torch
    torch.cuda.set_device(WORKER_GPU_ID)

def load_slot_masks(masks_pt_path: str) -> dict:
    """
    加载第一段代码生成的 _masks.pt 文件。

    返回:
        masks: [1, S, total_latent_tokens]
        config: num_frames, grid_h, grid_w, num_slots
    """
    data = torch.load(masks_pt_path, map_location="cpu")

    masks = data.get("masks")
    if masks is None:
        raise KeyError(f"{masks_pt_path} 中没有 key='masks'")

    masks = masks.float()

    # 兼容 [S, total_L] 和 [1, S, total_L]
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    elif masks.dim() != 3:
        raise ValueError(f"masks 维度应为 [S, L] 或 [1, S, L]，但得到 {tuple(masks.shape)}")

    config = {
        "num_frames": data.get("num_frames"),
        "grid_h": data.get("grid_h"),
        "grid_w": data.get("grid_w"),
        "num_slots": data.get("num_slots"),
    }

    missing = [k for k, v in config.items() if v is None]
    if missing:
        raise ValueError(f"_masks.pt 缺少必要字段: {missing}")

    return {
        "masks": masks,
        "masks_per_layer": data.get("masks_per_layer"),
        "config": config,
    }


def load_ground_truth(gt_pt_path: str) -> dict:
    data = torch.load(gt_pt_path, map_location="cpu")
    obj_masks = data["object_masks"]            # [T, O, H, W]，已过过滤
    if obj_masks.dtype != torch.float32:
        obj_masks = obj_masks.float()

    object_ids = data.get("object_ids", None)       # [T, O]，>=0 表示该帧该物体被认可
    active_matrix = data.get("active_matrix", None) # [T, O]，bool

    return {
        "object_masks":  obj_masks,
        "object_ids":    object_ids,
        "active_matrix": active_matrix,
        "height": data["height"],
        "width":  data["width"],
    }

def compute_fg_ari(assign, true_mask, K):
    B, T, H, W = assign.shape
    ari_per_frame = []
    for t in range(T):
        a_flat = assign[0, t].reshape(-1)                          # [N_px]
        tm     = true_mask[0, t]                                    # [O, H, W]
        tm_flat = einops.rearrange(tm, "o h w -> (h w) o")          # [N_px, O]
        fg = tm_flat.sum(-1) > 0
        if fg.sum() == 0:
            continue                                                # 跳过无前景帧
        pred_fg = F.one_hot(a_flat[fg], num_classes=K).float()
        true_fg = tm_flat[fg]
        ari_per_frame.append(
            adjusted_rand_index(true_fg.unsqueeze(0), pred_fg.unsqueeze(0)).item()
        )
    if not ari_per_frame:
        return 1.0
    return float(np.mean(ari_per_frame))


def compute_mbo(assign, true_mask, K):
    """
    true_mask: [B, T, O, H, W]，已经是"过滤后且仅保留认可像素"的 GT
    """
    B, T, H, W = assign.shape
    O = true_mask.shape[2]

    assign_flat = einops.rearrange(assign, "b t h w -> b (t h w)")
    true_flat   = einops.rearrange(true_mask, "b t o h w -> b (t h w) o")

    intersection = torch.zeros(B, K, O, device=true_flat.device, dtype=true_flat.dtype)
    for o in range(O):
        intersection[:, :, o].scatter_add_(
            dim=1, index=assign_flat, src=true_flat[:, :, o]
        )

    pred_area = torch.zeros(B, K, device=true_flat.device, dtype=true_flat.dtype)
    pred_area.scatter_add_(
        1, assign_flat, torch.ones_like(assign_flat, dtype=true_flat.dtype)
    )
    true_area = true_flat.sum(dim=1)                        # [B, O]

    union = pred_area[:, :, None] + true_area[:, None, :] - intersection
    iou   = intersection / union.clamp_min(1e-6)            # [B, K, O]
    best_per_obj = iou.max(dim=1).values                    # [B, O]

    # ---- 关键：只在物体真正激活的帧上平均 ----
    obj_per_frame = true_mask.any(dim=(3, 4))               # [B, T, O] bool
    # 每个物体在多少帧里是"存在"的
    frames_per_obj = obj_per_frame[0].sum(dim=0).float()    # [O]
    valid_obj = frames_per_obj > 0                          # [O]

    if valid_obj.sum() == 0:
        return 1.0

    # 逐物体 IoU：只在激活帧取平均
    obj_scores = []
    for o in range(O):
        if not valid_obj[o]:
            continue
        # best_per_obj 是按"全 T 的所有像素"一起算的 IoU，
        # 而我们要的是"只在物体出现过的帧上"的 IoU
        mask_o = obj_per_frame[0, :, o]                     # [T]
        inter_o = torch.zeros(K, device=true_flat.device)
        pred_area_o = torch.zeros(K, device=true_flat.device)
        true_area_o = 0.0
        for t in range(T):
            if not mask_o[t]:
                continue
            a = assign[0, t].reshape(-1)                    # [N_px]
            tm = true_mask[0, t, o].reshape(-1)             # [N_px]
            for k in range(K):
                sel = (a == k)
                inter_o[k]    += (tm[sel]).sum()
                pred_area_o[k] += sel.sum()
            true_area_o += tm.sum()
        union_o = pred_area_o + true_area_o - inter_o
        iou_o = inter_o / union_o.clamp_min(1e-6)
        obj_scores.append(iou_o.max().item())

    return float(np.mean(obj_scores))

def masks_to_assign(masks, grid_h, grid_w, target_h, target_w):
    """
    不 pad、不裁尾：
      - masks 里本来就有 collected_frames 帧（这里是 237）
      - 先在低分辨率 argmax，再 nearest 上采样到 (target_h, target_w)
        （避免 trilinear 把 slot 边界糊掉）
    返回:
      assign: [B, T_collected, target_h, target_w] int64
      S:      num_slots
      T_collected: 有效帧数
    """
    B, S, total_L = masks.shape
    tokens_per_frame = grid_h * grid_w
    if total_L % tokens_per_frame != 0:
        raise ValueError(
            f"total_L={total_L} 不能被 grid_h*grid_w={tokens_per_frame} 整除")
    collected_frames = total_L // tokens_per_frame

    masks_5d = masks.view(B, S, collected_frames, grid_h, grid_w).float()

    # 低分辨率 argmax：每个 token 归属哪个 slot
    assign_lr = masks_5d.argmax(dim=1)                 # [B, T, h, w]

    # nearest 上采样到目标分辨率
    assign_up = F.interpolate(
        assign_lr.float().unsqueeze(1),                # [B,1,T,h,w]
        size=(collected_frames, target_h, target_w),
        mode="nearest",
    )[:, 0].long()                                     # [B,T,H,W]

    return assign_up, S, collected_frames

def prepare_true_mask(gt_data, T_slot, skip_head, latent_to_pixel, device):
    """
    只做一次：把 GT 按 slot 时间轴对齐，并按 object_ids / active_matrix 过滤。
    返回： [1, T_slot, O, H_gt, W_gt] 的 float tensor（在 device 上）
    """
    obj_masks  = gt_data["object_masks"].to(device)     # [T_gt, O, H, W]
    object_ids = gt_data.get("object_ids")
    active_mat = gt_data.get("active_matrix")

    if object_ids is not None:
        object_ids = object_ids.to(device)
    if active_mat is not None:
        active_mat = active_mat.to(device)

    T_gt = obj_masks.shape[0]

    j = torch.arange(T_slot, device=device)
    gt_idx = (j + skip_head) * latent_to_pixel
    gt_idx = gt_idx.clamp(0, T_gt - 1).long()

    obj_masks_used = obj_masks[gt_idx]                  # [T_slot, O, H, W]

    # 只保留 ReferDINO 认可的物体实例
    if object_ids is not None:
        ids_used   = object_ids[gt_idx]                 # [T_slot, O]
        valid_obj  = (ids_used >= 0)
        obj_masks_used = obj_masks_used * valid_obj[:, :, None, None].float()

    # 只保留 prompt 计划里 active 的帧
    if active_mat is not None:
        act_used = active_mat[gt_idx]                   # [T_slot, O]
        obj_masks_used = obj_masks_used * act_used[:, :, None, None].float()

    return obj_masks_used.unsqueeze(0)                  # [1, T_slot, O, H, W]

def evaluate_layer_metrics(slot_data, layer, true_mask,
                           device="cuda") -> dict:
    """
    true_mask: [1, T_slot, O, H_gt, W_gt]，已经按 slot 时间轴对齐 + 过滤
               （由 prepare_true_mask 在外层统一构造一次）
    """
    cfg = slot_data["config"]

    if slot_data["masks_per_layer"] is not None and layer in slot_data["masks_per_layer"]:
        masks = slot_data["masks_per_layer"][layer].float()
    else:
        masks = slot_data["masks"]
    masks = masks.to(device)
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)

    target_h = true_mask.shape[3]
    target_w = true_mask.shape[4]

    # 低分辨率 argmax + nearest 上采样 → [B, T_slot, H, W]
    assign, K, T_slot = masks_to_assign(
        masks=masks,
        grid_h=cfg["grid_h"],
        grid_w=cfg["grid_w"],
        target_h=target_h,
        target_w=target_w,
    )

    assert T_slot == true_mask.shape[1], \
        f"T_slot 不一致：assign={T_slot}, true_mask={true_mask.shape[1]}"

    fg_ari = compute_fg_ari(assign, true_mask, K)
    mbo    = compute_mbo(assign, true_mask, K)

    del assign, masks
    return {"layer": layer, "fg_ari": fg_ari, "mbo": mbo}

def process_one_video(args):
    """
    处理单个视频，返回 (prompt_id, seed_id, layer_metrics) 或 None。
    """
    mask_path, gt_base_dir, skip_head = args
    import os, re, torch

    basename = os.path.basename(mask_path)
    prefix   = basename.replace("_masks.pt", "")

    m = re.match(r"rank(\d+)-(\d+)-(\d+)_lora", prefix)
    if m is None:
        return None

    rank_id, prompt_id, seed_id = m.group(1), m.group(2), m.group(3)

    gt_path = os.path.join(gt_base_dir, prefix, "referdino_objects.pt")
    if not os.path.exists(gt_path):
        return None

    device = f"cuda:{WORKER_GPU_ID}"
    latent_to_pixel = 4

    slot_data = load_slot_masks(mask_path)
    gt_data   = load_ground_truth(gt_path)

    # ---- 1) 先推 T_slot（所有层都一样） ----
    cfg = slot_data["config"]
    gh, gw = cfg["grid_h"], cfg["grid_w"]
    if slot_data["masks_per_layer"] is not None:
        first = next(iter(slot_data["masks_per_layer"].values()))
        if first.dim() == 3:      # [1, S, L]
            total_L = first.shape[-1]
        else:                     # [S, L]
            total_L = first.shape[-1]
    else:
        total_L = slot_data["masks"].shape[-1]
    T_slot = total_L // (gh * gw)

    # ---- 2) 只构造一次 true_mask（时间对齐 + object_ids / active 过滤） ----
    true_mask = prepare_true_mask(
        gt_data,
        T_slot=T_slot,
        skip_head=skip_head,
        latent_to_pixel=latent_to_pixel,
        device=device,
    )   # [1, T_slot, O, H_gt, W_gt]，在 GPU 上

    # 释放 CPU 上的原始 GT（true_mask 已经拷到 GPU）
    del gt_data["object_masks"]
    if "object_ids" in gt_data:    del gt_data["object_ids"]
    if "active_matrix" in gt_data: del gt_data["active_matrix"]

    # ---- 3) 逐层评估，复用同一份 true_mask ----
    if slot_data["masks_per_layer"] is not None:
        layers = sorted(int(k) for k in slot_data["masks_per_layer"].keys())
    else:
        layers = [15]

    layer_metrics = {}
    for layer in layers:
        result = evaluate_layer_metrics(
            slot_data, layer,
            true_mask=true_mask,
            device=device,
        )
        layer_metrics[layer] = {
            "fg_ari": float(result["fg_ari"]),
            "mbo":    float(result["mbo"]),
        }

    del slot_data, gt_data, true_mask
    torch.cuda.empty_cache()

    return prompt_id, seed_id, layer_metrics

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    import glob
    import json
    import re
    import numpy as np

    masks_dir   = "videos/interactive_train_long_slot_addchangeloss_ocvtp_entitybankfix_newreconstage_benchmark"
    gt_base_dir = "third_party/ReferDINO/output/referdino_soci/entitybank"
    out_json    = "videos/iclr2027finalversion/entitybank_per_prompt_per_layer_metrics.json"
    skip_head   = 3

    mask_files = sorted(glob.glob(os.path.join(masks_dir, "*_masks.pt")))

    # ---------- 指定要使用的 GPU 列表 ----------
    gpu_ids = list(range(torch.cuda.device_count()))

    gpu_queue = mp.Queue()
    for gid in gpu_ids:
        gpu_queue.put(gid)

    args_list = [(mask_path, gt_base_dir, skip_head) for mask_path in mask_files]

    raw = {}

    # 用 tqdm 显示总进度，进程数 = GPU 数，每块 GPU 同时只跑一个视频
    with mp.Pool(processes=len(gpu_ids),
                 initializer=init_worker,
                 initargs=(gpu_queue,)) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_video, args_list),
            total=len(args_list),
            desc="Evaluating videos",
            ncols=100,
        ):
            if res is None:
                continue
            prompt_id, seed_id, layer_metrics = res
            raw.setdefault(prompt_id, {}).setdefault(seed_id, {})
            for layer, metrics in layer_metrics.items():
                raw[prompt_id][seed_id][layer] = metrics

    # ---------------- 保存原始结果（不分析） ----------------
    with open(out_json, "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)

    print(f"\n💾 已保存: {out_json}")