# SlotMemory: Object-Centric KV Memory for Streaming Long-Video Generation

<p align="center">
  <a href="https://arxiv.org/abs/2605.31033">
    <img src="https://img.shields.io/badge/arXiv-2605.31033-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://github.com/tj12323/SlotMemory">
    <img src="https://img.shields.io/github/stars/tj12323/SlotMemory?style=social" alt="GitHub stars">
  </a>
  <a href="https://huggingface.co/Weijiafudan/SlotMemory">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow" alt="Hugging Face">
  </a>
</p>

**Weijia Dou¹, Hui Li¹, Jiahao Cui¹, Lei Zhou², Jingdong Wang³, Siyu Zhu¹***

¹ Fudan University &nbsp;&nbsp; ² Meta Superintelligence Labs &nbsp;&nbsp; ³ Baidu

---

Table of Contents
=================

* [🔥 TL;DR](#-tldr)
* [✨ Highlights](#-highlights)
* [🛠️ Installation](#️-installation)
    * [Requirements](#requirements)
    * [Environment Setup](#environment-setup)
* [🧱 Checkpoint Preparation](#-checkpoint-preparation)
    * [Inference Checkpoints](#inference-checkpoints)
    * [Training Checkpoints](#training-checkpoints)
* [🚀 Quick Start](#-quick-start)
    * [Single-Prompt Video Generation](#single-prompt-video-generation)
    * [Interactive Long-Video Generation](#interactive-long-video-generation)
* [⚙️ Training](#️-training)
    * [Phase A: Causal Generator Initialization (Optional)](#phase-a-causal-generator-initialization-optional)
    * [Phase B: SlotMemory Long-Video Tuning](#phase-b-slotmemory-long-video-tuning)
        * [Stage 2.1: Frozen-Grouper Adaptation (1,000 Steps)](#stage-21-frozen-grouper-adaptation-1000-steps)
        * [Stage 2.2: End-to-End Slot Optimization (2,000 Steps)](#stage-22-end-to-end-slot-optimization-2000-steps)
* [📏 Evaluation &amp; Benchmark](#-evaluation--benchmark)
    * [1. Short-Video and Long-Video Generation](#1-short-video-and-long-video-generation)
    * [2. Interactive Long-Video Generation](#2-interactive-long-video-generation)
    * [3. Segment-Wise Text Alignment](#3-segment-wise-text-alignment)
* [🤗 Acknowledgements](#-acknowledgements)
* [📄 Citation](#-citation)

## 🔥 TL;DR

**SlotMemory** introduces an object-centric Key-Value (KV) memory mechanism for streaming long-video generation.

Existing streaming video diffusion models typically organize historical context through temporal windows, compressed context tokens, or unstructured KV collections. Such temporal-centric memory representations make it difficult to preserve entity identity and semantic consistency over extended generation horizons, particularly when entities disappear from the scene or prompts change interactively.

SlotMemory shifts memory organization from **when information appears** to **what information represents**. It decomposes transformer KV representations into reusable semantic slots, which serve as routing addresses for selecting and retrieving original transformer KV tokens. Combined with a persistent entity bank and bounded memory assembly, SlotMemory enables entity-level memory retrieval while preserving the high-fidelity representations required by video diffusion.

On 60-second interactive video generation, SlotMemory achieves the highest Dynamic Degree (**81.38**) among evaluated streaming methods and the second-highest aggregate Quality Score (**81.61**). On 30-second generation, it achieves the best Total Score (**83.79**), Quality Score (**84.86**), and Dynamic Degree (**79.30**), improving Dynamic Degree by **31.2% over MemFlow**.

## ✨ Highlights

<p align="center">
  <img src="assets/pipeline.png" alt="Overview of SlotMemory" width="100%">
</p>

<p align="center">
  <em>
    Overview of SlotMemory. The framework integrates slot-based semantic KV compression,
    persistent entity memory, and bounded memory assembly for streaming long-video generation.
  </em>
</p>

- **Object-Centric KV Memory.** Decomposes historical transformer representations into semantic slots, enabling memory organization at the entity level rather than the temporal-frame level.
- **Original KV Preservation.** Uses semantic slots as routing addresses while retaining original transformer KV tokens as memory content, avoiding the information bottleneck of compressed latent representations.
- **Persistent Entity Bank.** Maintains entity identities, lifecycle states, and KV snapshots across streaming chunks, enabling retrieval of historical entity appearances beyond the local temporal window.
- **Prompt-Aware Memory Retrieval.** Preserves the entity bank and slot states across prompt transitions while refreshing text conditioning and recent local context under new instructions.
- **Bounded Memory Assembly.** Selectively retrieves relevant entity snapshots and historical KV groups under a fixed memory budget, supporting extended generation horizons.

---

## 🛠️ Installation

### Requirements

- Linux
- Python 3.10
- NVIDIA GPU with 80 GB memory (H200 tested)
- CUDA 12.4-compatible PyTorch environment

### Environment Setup

Clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/tj12323/SlotMemory.git
cd SlotMemory
```

If you have already cloned the repository without submodules:

```bash
git submodule update --init --recursive
```

Create and activate the Conda environment:

```bash
conda create -n slotmemory python=3.10 -y
conda activate slotmemory
```

Install PyTorch and the required dependencies:

```bash
python -m pip install "setuptools<81"
pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# FlashAttention
pip install flash-attn --no-build-isolation

# SlotContrast
pip install --no-deps -e third_party/slotcontrast
```

---

## 🧱 Checkpoint Preparation

We provide pretrained SlotMemory checkpoints through Hugging Face.

### Inference Checkpoints

**1. Wan2.1-T2V-1.3B**

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir wan_models/Wan2.1-T2V-1.3B
```

**2. SlotMemory**

```bash
hf download Weijiafudan/SlotMemory \
    --local-dir checkpoints
```

### Training Checkpoints

**1. Wan2.1-T2V-14B Teacher**

Download the teacher model:

```bash
hf download Wan-AI/Wan2.1-T2V-14B \
    --local-dir wan_models/Wan2.1-T2V-14B
```

**2. Causal Generator Initialization**

The initial causal generator is obtained through the Self-Forcing distribution matching distillation pipeline.

Follow [Self-Forcing](https://github.com/guandeh17/Self-Forcing) to prepare the VidProM training prompts and ODE-initialized checkpoint.

Alternatively, you can directly use the LongLive first-stage checkpoint:

```text
checkpoints/longlive_base.pt
```

This allows you to skip the initial causal generator initialization stage.

**3. Slot Initialization**

The reported model initializes its slot modules using a pretrained temporally consistent Slot Attention checkpoint from [SlotContrast](https://github.com/martius-lab/slotcontrast.git). Make the corresponding initialization checkpoint available at the path specified in the training configuration.

---

## 🚀 Quick Start

After installation and checkpoint preparation, you can directly run the pretrained model.

### Single-Prompt Video Generation

```bash
bash inference.sh
```

### Interactive Long-Video Generation

```bash
bash interactive_inference.sh
```

The interactive pipeline supports sequential prompt transitions while retaining the persistent entity memory across the entire generation process.

---

## ⚙️ Training

SlotMemory training consists of two phases:

- **Phase A:** Causal generator initialization through Self-Forcing.
- **Phase B:** SlotMemory long-video tuning through streaming rollout and distribution matching distillation.

### Phase A: Causal Generator Initialization (Optional)

This phase initializes the causal Wan2.1-T2V-1.3B generator.

Prepare the VidProM prompts and ODE-initialized checkpoint following [Self-Forcing](https://github.com/guandeh17/Self-Forcing).

**Single-node training:**

```bash
bash train_init.sh
```

**Multi-node distributed training:**

```bash
bash train_init_multinode.sh
```

If you already have the LongLive first-stage checkpoint (`checkpoints/longlive_base.pt`), you can skip this phase and proceed directly to Phase B.

### Phase B: SlotMemory Long-Video Tuning

The SlotMemory memory modules are optimized through self-generated streaming rollouts and distribution matching distillation.

The full training schedule consists of 3,000 updates, divided into two stages.

#### Stage 2.1: Frozen-Grouper Adaptation (1,000 Steps)

Set the following configuration:

```yaml
freeze_slot_grouper: true
```

This stage freezes the slot grouper, slot initializer, and dynamics predictor, allowing the input/output adapters and diffusion-side LoRA parameters to adapt to a fixed slot vocabulary.

Only the temporal contrastive slot loss is enabled during this stage; reconstruction and selection distillation losses are introduced in the subsequent stage.

Run:

```bash
bash train_long.sh
```

Train for **1,000 steps**.

#### Stage 2.2: End-to-End Slot Optimization (2,000 Steps)

Resume training from the Stage 2.1 checkpoint.

Update the configuration:

```yaml
freeze_slot_grouper: false
```

Additionally, disable the 1,000-step early-stopping condition in `trainer/distillation.py` so that training can continue beyond the Stage 2.1 boundary.

Run:

```bash
bash train_long.sh
```

Train for another **2,000 steps**. During this stage, the slot grouper, initializer, and dynamics predictor are jointly optimized with the remaining memory modules. All three slot regularization objectives are enabled.

---

## 📏 Evaluation & Benchmark

We evaluate SlotMemory across four generation horizons and two distinct generation scenarios:

| Scenario | Duration | Evaluation Protocol |
|:---|:---|:---|
| Short-video generation | 5s | Single prompt |
| Long-video generation | 30s | Single prompt |
| Long-horizon generation | 120s / 240s | Single prompt |
| Interactive long-video generation | 60s | Six prompts, five transitions |

All quantitative comparisons follow the evaluation protocols described in the paper.

### 1. Short-Video and Long-Video Generation

For 5s, 30s, 120s, and 240s generation, we follow the evaluation protocols adopted by [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [LongLive](https://github.com/NVlabs/LongLive).

Use the extended evaluation prompts provided by Self-Forcing:

```text
prompts/all_dimension_extended.txt
```

Generate videos using the single-prompt inference pipeline:

```bash
bash inference.sh
```

The generated videos are evaluated using:

- [VBench](https://github.com/Vchitect/VBench) for short-video generation.
- VBench-Long for long-horizon generation.

### 2. Interactive Long-Video Generation

We adopt the interactive benchmark provided by [MemFlow](https://github.com/KlingTeam/MemFlow).

The benchmark contains 100 narrative scripts, each consisting of six consecutive 10-second prompts. The model must maintain consistent entity appearance and semantic relationships across five prompt transitions.

The benchmark file is:

```text
interactive_benchmark.jsonl
```

Generate interactive videos using:

```bash
bash interactive_inference.sh
```

### 3. Segment-Wise Text Alignment

To evaluate prompt adherence throughout interactive generation, we compute text alignment independently for each 10-second segment.

Run:

```bash
bash clip_score/calclip.sh
```

We use **Qwen3-VL-Embedding-8B** to evaluate text alignment between generated video segments and their corresponding prompts.

The resulting scores provide a segment-wise assessment of prompt adherence across the complete 60-second generation sequence.

---

## 🤗 Acknowledgements

We sincerely thank the following projects and their authors for their excellent research and open-source contributions:

- [LongLive](https://github.com/NVlabs/LongLive): The streaming video generation codebase that SlotMemory builds upon.
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing): The distribution matching distillation framework used for causal generator initialization and training.
- [Wan2.1](https://github.com/Wan-Video/Wan2.1): The pretrained video diffusion backbone used in our work.
- [MemFlow](https://github.com/KlingTeam/MemFlow): The interactive long-video benchmark adopted in our evaluation.

We are grateful to the authors and contributors for making their work publicly available.

---

## 📄 Citation

If you find SlotMemory useful for your research, please consider citing our paper and giving this repository a star.

```bibtex
@misc{dou2026slotmemory,
    title={SlotMemory: Object-Centric KV Memory for Streaming Long-Video Generation},
    author={Weijia Dou and Hui Li and Jiahao Cui and Lei Zhou and Jingdong Wang and Siyu Zhu},
    year={2026},
    eprint={2605.31033},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2605.31033}
}
```
