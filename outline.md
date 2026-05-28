## 1. Hero

**Goal:** Show the result immediately.

**Title:**
**SlotMemory: Object-Centric KV Memory for Streaming Long-Video Generation**

**Tagline:**
**Remember objects, not frames.**

**Content:**

- One strong 60-second demo or a 3-row teaser strip.
- Short one-sentence summary.
- Buttons: Paper, Code, Checkpoints, BibTeX.
- Metric strip:
  - **81.61** 60s Quality
  - **74.29** 30s Dynamic Score
  - **22.8%** Dynamic Consistency Gain
  - **60s** Interactive Narratives

**Keep it cinematic and simple.**

---

## 2. Method Overview

**Title:**
**A read-write-update loop for streaming video**

**Core message:**
For every chunk, SlotMemory retrieves relevant memories, generates the next chunk, writes new slot memories, and evicts obsolete ones.

**Content:**

Use four steps:

1. **Read** relevant slot memories.
2. **Generate** the current video chunk.
3. **Write** new slot-conditioned KV memories.
4. **Update** the memory bank under a fixed budget.

**Visual:**
A cleaned-up version of the pipeline diagram:

```text
Current Chunk
   ↓
Read Slot Memory
   ↓
Denoising Transformer
   ↓
Write New Slots
   ↓
Update Memory Bank
   ↓
Next Chunk
```

**Optional interaction:**
A small animated loop is enough. Avoid multi-tab technical panels.

---

## 3. Results

**Title:**
**Stronger consistency over longer videos**

**Core message:**
SlotMemory improves most where memory matters most: 30–60 second generation.

### 3.1 60-second interactive generation

Use one compact table:

| Method         | Quality ↑ | Avg. CLIP ↑ |
| -------------- | --------: | ----------: |
| Infinity-RoPE  |     79.98 |       22.31 |
| LongLive       |     79.09 |       25.14 |
| MemFlow        |     78.57 |       24.09 |
| **SlotMemory** | **81.61** |   **25.60** |

The Avg. CLIP is the average of the six reported 10-second CLIP segments.

### 3.2 30-second long-video generation

Highlight:

- **74.29 Dynamic Score**
- **84.28 Total Score**
- **85.23 Quality Score**

### 3.3 5-second preservation

Highlight:

- SlotMemory preserves short-video quality.
- The memory module does not degrade base generation.

**Visual:**
Use three metric cards, not several large tables.

---

## 4. Qualitative Demos

**Title:**
**Long-form examples**

**Core message:**
SlotMemory preserves subject identity, props, and spatial relations across scene transitions.

**Content:**

Use 3–6 strong demo rows:

1. Animal identity persistence
2. Human-object interaction
3. Vehicle or rider continuity
4. Prompt-switch narrative
5. Occlusion and re-entry
6. Failure case

Each demo should include:

- video or keyframe strip,
- prompt timeline,
- one-sentence takeaway.

**Avoid:**
Too many method selectors, complex overlays, or heavy comparison controls.

---

## 5. Citation

**Title:**
**Citation**

Include BibTeX and acknowledgments.
