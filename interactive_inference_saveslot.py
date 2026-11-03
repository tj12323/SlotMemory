# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import argparse
import os
from typing import List
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from torchvision import transforms  # noqa: F401
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier  
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

from pipeline.interactive_causal_inference import (
    InteractiveCausalInferencePipeline,
)
# from pipeline.harvest_pipeline import get_real_model
from utils.dataset import MultiTextDataset
import json

import inspect  # 放到文件顶部 import 区

collected_masks = {}
is_recache_active = False
current_step_meta = {"t": None, "current_start": None, "cache_start": None}
call_count = 0

# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Interactive causal inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    
    # Set device first
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group with backend and timeout
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )
    
    # set_seed(config.seed + local_rank)
    set_seed(config.seed)
    print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    print(f"Single GPU mode on device {device}")


low_memory = get_cuda_free_memory_gb(device) < 40
torch.set_grad_enabled(False)

pipeline = InteractiveCausalInferencePipeline(config, device=device)

if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]

    if config.use_ema:
        def _clean_key(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "")

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(
            cleaned_state_dict, strict=False
        )
        if local_rank == 0:
            if missing:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if unexpected:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        # 加载主模型权重
        missing, unexpected = pipeline.generator.load_state_dict(raw_gen_state_dict, strict=False)
        print(f"Main model loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        print(f"Missing keys ({len(missing)}): {missing[:20]}")
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}")

# # 我们以监控第 15 层为例
# target_layer_idx = 5
# probe_layer = pipeline.generator.model.blocks[target_layer_idx].self_attn

# # 开启监控开关
# probe_layer.record_attention_stats = True

# # 动态指定日志文件名（带上当前的 bank_size 参数，方便后续对比）
# current_bank_size = config.model_kwargs.get('bank_size', 3)
# probe_layer.attn_log_path = f"results/attn_stats_layer{target_layer_idx}_b{current_bank_size}.jsonl"

# # 如果文件已存在，先清空它防止数据追加混乱
# if os.path.exists(probe_layer.attn_log_path):
#     os.remove(probe_layer.attn_log_path)
    
# print(f"Attention probe activated on Layer {target_layer_idx}, logging to {probe_layer.attn_log_path}")
def _load_named_module_state_dict(root_model, module_name, state_dict, strict=True,is_main_process=False):
        """
        Load a state_dict into a named submodule.

        Args:
            root_model: non-FSDP inner model root, e.g. self.model.generator.model
            module_name: submodule path relative to root_model, e.g. "memory_compressor"
            state_dict: state dict relative to that submodule
        """
        module = root_model
        for attr in module_name.split("."):
            module = getattr(module, attr)

        result = module.load_state_dict(state_dict, strict=strict)

        if is_main_process:
            print(f"[Info] Loaded state_dict into {module_name}")
            if hasattr(result, "missing_keys") and len(result.missing_keys) > 0:
                print(f"  missing_keys: {result.missing_keys}")
            if hasattr(result, "unexpected_keys") and len(result.unexpected_keys) > 0:
                print(f"  unexpected_keys: {result.unexpected_keys}")
                
def _load_memory_gate_mlp_state_dict(root_model, state_dict_map, strict=True, is_main_process=False):
    """
    Load per-block memory_gate_mlp state_dicts into the non-FSDP generator model.

    Args:
        root_model: non-FSDP inner model root, e.g. pipeline.generator.model
        state_dict_map: dict like
            {
                "blocks.0.self_attn.memory_gate_mlp": {...},
                "blocks.1.self_attn.memory_gate_mlp": {...},
                ...
            }
    """
    if state_dict_map is None:
        if is_main_process:
            print("[Warning] generator_memory_gate_mlp is None, skip loading.")
        return

    loaded_blocks = 0
    total_tensors = 0

    for module_name, module_sd in state_dict_map.items():
        module = root_model
        for attr in module_name.split("."):
            if attr.isdigit():
                module = module[int(attr)]
            else:
                module = getattr(module, attr)

        result = module.load_state_dict(module_sd, strict=strict)
        loaded_blocks += 1
        total_tensors += len(module_sd)

        if is_main_process:
            print(f"[Info] Loaded state_dict into {module_name}")
            if hasattr(result, "missing_keys") and len(result.missing_keys) > 0:
                print(f"  missing_keys: {result.missing_keys}")
            if hasattr(result, "unexpected_keys") and len(result.unexpected_keys) > 0:
                print(f"  unexpected_keys: {result.unexpected_keys}")

    if is_main_process:
        print(f"[Info] Loaded memory_gate_mlp for {loaded_blocks} blocks, total tensors: {total_tensors}")


# ========== 新增：Monkey Patch 函数 ==========
def make_patched_forward_inference(original_fn):
    # 拿到未绑定的原始函数，以便调用时自己传 self
    orig_func = original_fn.__func__
    orig_sig = inspect.signature(orig_func)

    def patched_fn(self, *args, **kwargs):
        # import pdb;pdb.set_trace()
        global is_recache_active, current_step_meta, call_count

        # 用 signature 解析所有参数，最可靠
        try:
            bound = orig_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            params = bound.arguments
            t = params.get("t")
            current_start = params.get("current_start")
            cache_start = params.get("cache_start")
            is_recache = params.get("is_recache", False)
        except TypeError as e:
            print(f"[WARN] bind failed in patched _forward_inference: {e}")
            t = current_start = cache_start = None
            is_recache = False

        is_recache_active = is_recache
        call_count += 1

        # t 可能是 tensor，转成标量方便后续比较
        if torch.is_tensor(t):
            t_val = t.flatten()[0].item()
        else:
            t_val = t

        current_step_meta = {
            "t": t_val,
            "current_start": current_start,
            "cache_start": cache_start,
        }

        # # 打印调用次数、t、current_start、is_recache，方便确认
        # if not is_recache:
        #     print(f"[DEBUG] call={call_count} t={t_val} "
        #           f"current_start={current_start} cache_start={cache_start} "
        #           f"is_recache={is_recache}")

        return orig_func(self, *args, **kwargs)

    return patched_fn

def make_monkey_patch_forward(original_forward):
    def patched_forward(self, new_k, new_v, layer_idx, query):
        global is_recache_active, current_step_meta
        out_dict = original_forward(new_k, new_v, layer_idx, query)
        if "masks" in out_dict:
            masks = out_dict["masks"].detach().cpu()
            if not is_recache_active:
                if layer_idx not in collected_masks:
                    collected_masks[layer_idx] = []
                collected_masks[layer_idx].append({
                    "masks": masks,
                    "t": current_step_meta.get("t"),
                    "current_start": current_step_meta.get("current_start"),
                    "cache_start": current_step_meta.get("cache_start"),
                })
        return out_dict
    return patched_forward
# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # After loading base weights, apply LoRA wrapper to the generator's transformer model
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights (if lora_ckpt is provided)
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # Support both formats: containing the `generator_lora` key or a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")
    
    # 2. Load memory_compressor full weights
    if isinstance(lora_checkpoint, dict) and "generator_memory_compressor" in lora_checkpoint:
        if local_rank == 0:
            print(f"Loading generator memory_compressor weights: {len(lora_checkpoint['generator_memory_compressor'])} tensors")
        _load_named_module_state_dict(
            pipeline.generator.model,
            "memory_compressor",
            lora_checkpoint["generator_memory_compressor"],
            strict=True,
            is_main_process=(local_rank == 0),
        )
    else:
        if local_rank == 0:
            print("Warning: generator_memory_compressor not found in LoRA checkpoint.")
    
    # 3. Load memory_gate_mlp full weights
    if isinstance(lora_checkpoint, dict) and "generator_memory_gate_mlp" in lora_checkpoint:
        if local_rank == 0:
            print(
                f"Loading generator memory_gate_mlp weights for "
                f"{len(lora_checkpoint['generator_memory_gate_mlp'])} blocks"
            )
        _load_memory_gate_mlp_state_dict(
            pipeline.generator.model,
            lora_checkpoint["generator_memory_gate_mlp"],
            strict=True,
            is_main_process=(local_rank == 0),
        )
    else:
        if local_rank == 0:
            print("Warning: generator_memory_gate_mlp not found in LoRA checkpoint.")

    pipeline.is_lora_enabled = True

# ========== 新增：应用 Monkey Patch ==========
model = pipeline.generator.model
if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
    model = model.base_model.model   # PeftModel -> 底层 CausalWanModel

model._forward_inference = make_patched_forward_inference(
    model._forward_inference
).__get__(model, model.__class__)
print("🐒 Successfully monkey-patched CausalWanModel._forward_inference to track is_recache!")

if getattr(pipeline.generator.model, "memory_compressor", None) is not None:
    mc = pipeline.generator.model.memory_compressor
    mc.forward = make_monkey_patch_forward(mc.forward).__get__(mc, mc.__class__)
    print("🐒 Successfully monkey-patched MemoryCompressor.forward to collect attention masks!")
else:
    print("Error: MemoryCompressor is not mounted. Check slot settings in config.")
    sys.exit(1)

# Move pipeline to appropriate dtype and device
print("dtype", pipeline.generator.model.dtype)
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- Build dataset -----------------------------
# Parse switch_frame_indices
switch_frame_indices: List[int] = [int(x) for x in config.switch_frame_indices.split(",") if x.strip()]

# Create dataset
dataset = MultiTextDataset(config.data_path)

# Validate number of segments & switch_frame_indices length
num_segments = len(dataset[0]["prompts_list"])
assert len(switch_frame_indices) == num_segments - 1, (
    "The number of switch_frame_indices should be the number of prompt segments minus 1"
)

print("Number of segments:", num_segments)
print("Switch frame indices:", switch_frame_indices)

num_prompts_total = len(dataset)
print(f"Number of prompt lines: {num_prompts_total}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ----------------------------- Inference loop -----------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    prompts_list: List[str] = batch_data["prompts_list"]  # type: ignore

    # === 新增：外部循环生成 5 个样本 ===
    num_samples_needed = config.num_samples  # 你想要的样本数，可改成 config.num_samples
    # ===== 提前计算 model_type（只需一次，放在外循环开始前亦可） =====
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"

    for seed_idx in range(num_samples_needed):
        torch.manual_seed(config.seed + seed_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed + seed_idx)

        # ✅ 新增：构造预期视频路径并检查是否已存在
        if config.save_with_index:
            video_path_check = os.path.join(
                config.output_folder,
                f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4"
            )
        else:
            short_name = prompts_list[0][0][:100].replace("/", "_")
            video_path_check = os.path.join(
                config.output_folder,
                f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4"
            )

        if os.path.exists(video_path_check):
            print(f"[Rank {rank}] Video already exists, skipping seed {seed_idx}: {video_path_check}")
            collected_masks.clear()        # 避免残留数据影响后续
            continue                       # ⬅️ 跳过生成

        sampled_noise = torch.randn(
            [1, config.num_output_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16,
        )

        # ========== 清空收集容器 ==========
        collected_masks.clear()

        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
        )

        # ========== 保存视频 + masks 数据 (仅 rank 0) ==========        
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

        # 确定模型类型（与原有逻辑一致）
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"

        # 视频保存路径 (沿用原有逻辑)
        if config.save_with_index:
            if config.num_samples==1:
                seed_idx_save = config.seed
            else:
                seed_idx_save = seed_idx
            video_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx_save}_{model_type}.mp4")
        else:
            # Use the first prompt segment as the filename prefix to avoid overly long names
            short_name = prompts_list[0][0][:100].replace("/", "_")
            video_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")

        write_video(video_path, current_video[0].to(torch.uint8), fps=16)

        # ========== 保存 masks 数据 ==========
        target_layer = getattr(config, "target_layer", 15)
        if target_layer in collected_masks:
            from collections import Counter
            raw_entries = collected_masks[target_layer]

            # 统计各去噪步 t 的 entry 数
            t_counter = Counter(e["t"] for e in raw_entries)
            all_ts = sorted(t_counter.keys())
            last_t = all_ts[0]   # 默认最后一次去噪步 t 最小；若采样器 t 递增，改用 all_ts[-1]

            selected_entries = [e for e in raw_entries if e["t"] == last_t]
            selected_entries = sorted(
                selected_entries,
                key=lambda e: e["current_start"] if e["current_start"] is not None else 0
            )

            masks_list = [e["masks"] for e in selected_entries]
            all_masks = torch.cat(masks_list, dim=-1)   # [S, total_tokens]

            # 帧数对齐（参考第二段 process_and_save_visualizations）
            grid_h = config.model_kwargs.get("spatial_grid_h", 30)
            grid_w = config.model_kwargs.get("spatial_grid_w", 52)
            tokens_per_frame = grid_h * grid_w
            total_collected_latent_frames = all_masks.shape[-1] // tokens_per_frame

            # 如果收集到的 latent 帧数超过 num_output_frames，
            # 说明开头包含上一个 chunk 的 tail，只保留最后 num_output_frames 帧
            if total_collected_latent_frames > config.num_output_frames:
                all_masks = all_masks[:, -config.num_output_frames * tokens_per_frame:]

            pt_path = video_path.replace(".mp4", "_masks.pt")
            save_data = {
                "masks": all_masks,
                "prompts_list": prompts_list,
                "switch_frame_indices": switch_frame_indices,
                "num_frames": config.num_output_frames,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "num_slots": config.model_kwargs.get("num_slots_per_chunk", 7),
            }

            # masks_per_layer 也要同步改成：按最后一步筛选 + 按 current_start 排序
            save_data["masks_per_layer"] = {}
            for k, entries in collected_masks.items():
                t_counter_k = Counter(e["t"] for e in entries)
                last_t_k = sorted(t_counter_k.keys())[0]
                sel_k = sorted(
                    [e for e in entries if e["t"] == last_t_k],
                    key=lambda e: e["current_start"] if e["current_start"] is not None else 0
                )
                save_data["masks_per_layer"][k] = torch.cat([e["masks"] for e in sel_k], dim=-1)

            torch.save(save_data, pt_path)
            print(f"✅ Saved masks to {pt_path}")
        else:
            print(f"⚠️  Layer {target_layer} not captured. Available: {list(collected_masks.keys())}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group() 