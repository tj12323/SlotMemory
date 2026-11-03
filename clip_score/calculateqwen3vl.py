import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
import decord
from decord import VideoReader
from utils.dataset import MultiTextDataset  # type: ignore

# 引入原始 embedder，用来继承
from qwen3_vl_embedding import Qwen3VLEmbedder, sample_frames as original_sample_frames

# ---------------------------------------------------------------------------
# 优化 1：修改模型初始化 (inference_mode, flash_attn, bf16, TF32, 降低像素)
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class Qwen3VLEmbedderOptimized(Qwen3VLEmbedder):
    """
    继承原始 Qwen3VLEmbedder 并重写核心方法：
    - forward 使用 @torch.inference_mode() + use_cache=False
    - process 使用 @torch.inference_mode()，保持 Tensor 在 GPU
    - 修改默认 max_frames=None，避免上采样短帧序列
    - 降低 default_total_pixels 控制视频总 token 预算
    """

    def __init__(self, *args, **kwargs):
        # 设置合理的视频总像素预算（约 4 帧 256×256 的 token 量）
        kwargs.setdefault("total_pixels", 4 * 256 * 32 * 32)  # 4 帧 * 每帧 256 tokens
        kwargs.setdefault("max_frames", None)                 # 不强制帧数
        kwargs.setdefault("num_frames", None)
        super().__init__(*args, **kwargs)
        self.model.eval()

    @torch.inference_mode()
    def forward(self, inputs: Dict) -> Dict[str, torch.Tensor]:
        # 关键：显式关闭 KV cache（embedding 场景不需要）
        outputs = self.model(**inputs, use_cache=False)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    @torch.inference_mode()
    def process(
        self,
        inputs: List[Dict[str, any]],
        normalize: bool = True,
    ) -> torch.Tensor:
        # 使用父类的 format_model_input，但注意我们已经改写了 max_frames 等
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                video=ele.get("video"),
                instruction=ele.get("instruction"),
                fps=None,   # 不指定 fps，避免内部帧采样
                max_frames=None,
            )
            for ele in inputs
        ]
        processed_inputs = self._preprocess_inputs(conversations)
        processed_inputs = {
            k: v.to(self.model.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in processed_inputs.items()
        }
        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        return embeddings


# ---------------------------------------------------------------------------
# 其他辅助函数保持不变
# ---------------------------------------------------------------------------
TEMPORAL_COMPRESSION = 4


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-VL-Embedding-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sample_fps: float = 1.0  # 此处保留，但原生视频 embedding 不使用


def parse_idx_from_filename(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"rank\d+-(\d+)-\d+_", base)
    return int(m.group(1)) if m else None

def sanitize_filename(text, max_length=100):
    """Remove or replace invalid filename characters."""
    # Replace invalid characters with underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', text)
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    # Remove leading/trailing underscores and dots
    sanitized = sanitized.strip('_.')
    # Truncate to max_length
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized
def find_idx_by_prompt(video_path: str, dataset, seed=0) -> Optional[int]:
    """
    从视频文件名中提取 sanitized prompt，在 dataset 中查找对应的 idx。
    假设 dataset[idx]["prompts_list"][0] 就是生成视频所用的 prompt。
    """
    base = os.path.basename(video_path)
    # 移除后缀和末尾的 -{seed}，得到 safe_prompt
    base_no_ext = base.rsplit('.', 1)[0]  # 去掉 .mp4
    suffix = f"-{seed}"
    if base_no_ext.endswith(suffix):
        safe_prompt = base_no_ext[:-len(suffix)]
    else:
        return None  # 格式不符合预期

    # 遍历 dataset，寻找匹配
    for idx in range(len(dataset)):
        sample = dataset[idx]
        # 假设生成视频时用的是 prompts_list[0] 作为 prompt
        prompt = sample["prompts_list"][0]
        
        # 相同的清理函数
        candidate = sanitize_filename(prompt, max_length=100)
        if safe_prompt.startswith(candidate):
            return idx
    import pdb;pdb.set_trace()
    return None

# def parse_idx_from_filename(path: str) -> Optional[int]:
#     base = os.path.basename(path)
#     m = re.search(r"^(\d+)-\d+\.mp4$", base)
#     return int(m.group(1)) if m else None


def segment_frame_ranges(total_frames: int, switch_frame_indices: List[int]) -> List[Tuple[int, int]]:
    switches = [int(s) for s in switch_frame_indices]
    switches = [s for s in switches if 0 < s < total_frames]
    switches = sorted(set(switches))
    bounds = [0] + switches + [total_frames]
    ranges = []
    for i in range(len(bounds) - 1):
        start = bounds[i]
        end = bounds[i + 1] - 1
        if end < start:
            end = start
        ranges.append((start, end))
    return ranges


def sample_indices_uniform(start_f: int, end_f: int, max_frames: int = 4) -> List[int]:
    """每个 segment 固定采样 max_frames 帧（均匀取），减少计算量"""
    if end_f <= start_f:
        return [start_f]
    frame_count = end_f - start_f + 1
    n = min(frame_count, max_frames)
    return np.linspace(start_f, end_f, num=n, dtype=np.int64).tolist()


# ---------------------------------------------------------------------------
# 优化 3 & 4：全程 GPU tensor + 原生视频 embedding
# ---------------------------------------------------------------------------
def compute_video_segment_scores_native(
    video_path: str,
    prompts: List[str],
    switch_frame_indices: List[int],
    model: Qwen3VLEmbedderOptimized,
    prompt_cache: Dict[str, torch.Tensor],   # GPU tensor dict
    max_frames_per_segment: int = 4,
) -> Dict:
    """
    使用原生视频 embedding 对每个 segment 进行编码：
    - 每个 segment 采样 max_frames_per_segment 帧
    - 将 segment 的帧列表作为 video 输入，一次 batch 编码所有 segment
    - 与提前编码好的文本 embedding 在 GPU 上计算点积
    """
    decord.bridge.set_bridge("native")
    vr = VideoReader(video_path)
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 16.0

    # 计算 switch 对应的像素帧索引
    switch_indices_pixel = [int(s) * TEMPORAL_COMPRESSION for s in switch_frame_indices]
    ranges = segment_frame_ranges(total_frames, switch_indices_pixel)

    if len(ranges) != len(prompts):
        raise ValueError(
            f"Segment mismatch for {os.path.basename(video_path)}: "
            f"{len(ranges)} ranges vs {len(prompts)} prompts."
        )

    # 对每个 segment 采样帧
    seg_frame_ids = []
    all_frame_ids = []
    for fr0, fr1 in ranges:
        ids = sample_indices_uniform(fr0, fr1, max_frames_per_segment)
        seg_frame_ids.append(ids)
        all_frame_ids.extend(ids)

    # 读取所有帧
    if all_frame_ids:
        frame_arr = vr.get_batch(all_frame_ids).asnumpy()
        pil_frames = [Image.fromarray(f) for f in frame_arr]
    else:
        pil_frames = []

    # 按 segment 分组 PIL 帧列表
    cursor = 0
    seg_video_frames = []
    for ids in seg_frame_ids:
        n = len(ids)
        seg_video_frames.append(pil_frames[cursor:cursor + n])
        cursor += n

    # --- 原生视频 embedding：一次 process 所有 segment ---
    if seg_video_frames and any(len(f) > 0 for f in seg_video_frames):
        segment_inputs = [{"video": frames} for frames in seg_video_frames]
        segment_embs = model.process(segment_inputs, normalize=True)  # (num_seg, dim), GPU
    else:
        segment_embs = torch.empty(0, self.hidden_size, device=model.model.device)

    # --- 文本 embedding：从缓存取，已在 GPU 上 ---
    text_embs = torch.stack([prompt_cache[p] for p in prompts], dim=0).float()

    # --- 余弦相似度（点积，因均已 L2 norm）---
    scores = torch.sum(segment_embs * text_embs, dim=-1)  # (num_seg,)

    # 构建返回结果
    per_segment = []
    for i, (ids, score) in enumerate(zip(seg_frame_ids, scores.cpu().tolist())):
        start_s = ids[0] / fps if ids else 0.0
        end_s = (ids[-1] + 1) / fps if ids else 0.0
        per_segment.append({
            "segment_idx": i,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "num_frames": len(ids),
            "qwen_score": score,
        })

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "per_segment": per_segment,
        "video_mean": float(np.mean([r["qwen_score"] for r in per_segment]) if per_segment else 0.0),
    }


# ---------------------------------------------------------------------------
# 优化 2：批量预计算所有唯一 prompt
# ---------------------------------------------------------------------------
@torch.inference_mode()
def precompute_prompt_embeddings(
    model: Qwen3VLEmbedderOptimized,
    prompts: List[str],
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """批量编码所有唯一 prompt，返回 GPU tensor 字典"""
    unique_prompts = list(dict.fromkeys(prompts))   # 保持顺序并去重
    cache: Dict[str, torch.Tensor] = {}
    for start in range(0, len(unique_prompts), batch_size):
        batch = unique_prompts[start:start + batch_size]
        embs = model.process([{"text": p} for p in batch], normalize=True).float()
        for prompt, emb in zip(batch, embs):
            cache[prompt] = emb.detach()  # 留在 GPU
    return cache


# ---------------------------------------------------------------------------
# Main 函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Batch Native Video Embedding Scoring (Optimized)")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-Embedding-8B")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_frames_per_segment", type=int, default=4,
                        help="每个 segment 均匀采样的最大帧数")
    parser.add_argument("--strict_idx_from_filename", type=int, default=1)
    parser.add_argument("--print_per_video", type=int, default=0)
    parser.add_argument("--prompt_batch_size", type=int, default=256,
                        help="预编码 prompt 的 batch size")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    dataset = MultiTextDataset(cfg.data_path)
    switch_frame_indices = [int(x) for x in str(cfg.switch_frame_indices).split(",") if str(x).strip()]
    if not switch_frame_indices:
        raise ValueError("config.switch_frame_indices is empty.")

    # 扫描所有视频，构建任务列表并收集所有 prompt
    video_paths = sorted(glob.glob(os.path.join(args.video_dir, args.pattern)))
    if not video_paths:
        raise FileNotFoundError(f"No videos found in {args.video_dir} with pattern {args.pattern}")

    jobs: List[Tuple[str, List[str]]] = []
    all_prompts: List[str] = []
    for vp in video_paths:
        if args.strict_idx_from_filename == 1:
            idx = parse_idx_from_filename(vp)
            if idx is None:
                raise ValueError(f"Cannot parse idx from filename: {os.path.basename(vp)}")
            # idx = find_idx_by_prompt(vp, dataset)  # 需要加上 --seed 参数
            # if idx is None:
            #     raise ValueError(f"Cannot find matching idx for video: {os.path.basename(vp)}")
        else:
            idx = 0
        sample = dataset[idx]
        prompts_list = list(sample["prompts_list"])
        jobs.append((vp, prompts_list))
        all_prompts.extend(prompts_list)

    # ---- 加载模型 (优化初始化) ----
    print(f"Loading model {args.model_name} on {args.device} ...")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    model = Qwen3VLEmbedderOptimized(
        model_name_or_path=args.model_name,
        dtype=dtype,
        attn_implementation="flash_attention_2",   # 或 "sdpa"
    )

    # ---- 批量预计算所有唯一 prompt (GPU tensor) ----
    print("Precomputing all unique prompt embeddings ...")
    prompt_cache = precompute_prompt_embeddings(model, all_prompts, batch_size=args.prompt_batch_size)
    print(f"  Cached {len(prompt_cache)} unique prompts on GPU.")

    # ---- 主循环：处理每个视频 ----
    segment_scores_acc: List[List[float]] = []
    segment_time_ref: Optional[List[Tuple[float, float]]] = None
    overall_all_segments_scores: List[float] = []

    for vp, prompts_list in tqdm(jobs, desc="Scoring videos"):
        if len(switch_frame_indices) != len(prompts_list) - 1:
            raise ValueError(
                f"Switch count mismatch: {len(switch_frame_indices)} switches vs "
                f"{len(prompts_list)} prompt segments."
            )

        out = compute_video_segment_scores_native(
            video_path=vp,
            prompts=prompts_list,
            switch_frame_indices=switch_frame_indices,
            model=model,
            prompt_cache=prompt_cache,
            max_frames_per_segment=args.max_frames_per_segment,
        )

        per_seg = out["per_segment"]
        if not segment_scores_acc:
            segment_scores_acc = [[] for _ in range(len(per_seg))]
            segment_time_ref = [(r["start_s"], r["end_s"]) for r in per_seg]
        if len(per_seg) != len(segment_scores_acc):
            raise ValueError("Segment count mismatch across videos")
        for i, r in enumerate(per_seg):
            segment_scores_acc[i].append(r["qwen_score"])
            overall_all_segments_scores.append(r["qwen_score"])

        if args.print_per_video == 1:
            print(f"\nVideo: {os.path.basename(vp)} | video_mean={out['video_mean']:.6f}")
            for r in per_seg:
                print(f"Seg {r['segment_idx']} [{r['start_s']:.1f}-{r['end_s']:.1f}s] "
                      f"frames={r['num_frames']}: {r['qwen_score']:.4f}")

    # ---- 汇总统计 ----
    print("\n========== Segment-wise Mean over ALL Videos ==========")
    assert segment_time_ref is not None
    for i, scores in enumerate(segment_scores_acc):
        mean_i = float(np.mean(scores) if scores else 0.0)
        s0, s1 = segment_time_ref[i]
        print(f"Seg {i} [{s0:.1f}-{s1:.1f}s] mean_over_videos={mean_i:.6f} (n={len(scores)})")

    global_all_segments_mean = float(np.mean(overall_all_segments_scores) if overall_all_segments_scores else 0.0)
    print("======================================================")
    print(f"Global mean over ALL segments of ALL videos: {global_all_segments_mean:.6f}")
    print("======================================================\n")


if __name__ == "__main__":
    main()