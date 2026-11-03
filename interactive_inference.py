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
import time
import numpy as np

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

latency_list = []
memory_list = []
# ----------------------------- Inference loop -----------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    prompts_list: List[str] = batch_data["prompts_list"]  # type: ignore

    # === 新增：外部循环生成 5 个样本 ===
    num_samples_needed = config.num_samples  # 你想要的样本数，可改成 config.num_samples
    for seed_idx in range(num_samples_needed):
        # 每次生成一个独立噪声（确保多样性）
        # 可选：固定不同种子让结果可复现
        torch.manual_seed(config.seed + seed_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed + seed_idx)
        sampled_noise = torch.randn(
            [1, config.num_output_frames, 16, 60, 104],  # batch 强制为 1
            device=device,
            dtype=torch.bfloat16,
        )        

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts_list=prompts_list,
            switch_frame_indices=switch_frame_indices,
            return_latents=False,
        )
        torch.cuda.synchronize(device)
        latency = time.perf_counter() - start
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        latency_list.append(latency)
        memory_list.append(peak_memory)

        # 后处理 & 保存（保持你原来的逻辑）
        current_video = rearrange(video, "b t c h w -> b t h w c").cpu() * 255.0

        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0

        # 确定模型类型（LoRA/EMA/regular）
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"

        if config.save_with_index:
            if config.num_samples==1:
                seed_idx_save = config.seed
            else:
                seed_idx_save = seed_idx
            output_path = os.path.join(config.output_folder, f"rank{rank}-{idx}-{seed_idx_save}_{model_type}.mp4")
        else:
            # Use the first prompt segment as the filename prefix to avoid overly long names
            short_name = prompts_list[0][0][:100].replace("/", "_")
            output_path = os.path.join(config.output_folder, f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4")

        write_video(output_path, current_video[0].to(torch.uint8), fps=16)  # 注意这里取 [0] 因为 batch=1

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if local_rank == 0:
    print("=" * 50)
    print(f"Average latency: {np.mean(latency_list):.3f} s")
    print(f"Latency std: {np.std(latency_list):.3f} s")
    print(f"Average peak VRAM: {np.mean(memory_list):.2f} GB")
    print(f"Maximum peak VRAM: {np.max(memory_list):.2f} GB")
    
if dist.is_initialized():
    dist.destroy_process_group() 