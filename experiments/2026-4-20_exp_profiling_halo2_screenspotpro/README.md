<!-- AI-GENERATED
     Model   : Claude Sonnet 4.6
     Date    : 2026-04-20
     Prompt  : write inside readme what profiling results mean and how to optimize them
-->

# Profiling — HALO2 on ScreenSpot-Pro

**Date:** 2026-04-20  
**Branch:** `aguvis-screenspot-pro`  
**Script:** `src/pipeline/profiling_grounder.py`  
**Config:** `profiling_grounder.yaml`

---

## Results

### Per-step timing (4 batches, batch_size=1)

| Step                   | Total (s) | Avg / batch (s) | % of predict time |
|------------------------|-----------|-----------------|-------------------|
| `preprocess_image`     | 1.00      | 0.25            | 0.3 %             |
| `preprocess_text`      | 0.03      | 0.007           | < 0.1 %           |
| `tokenize_and_transfer`| 7.95      | 1.99            | 2.6 %             |
| **`generation`**       | **309.07**| **77.27**       | **97.0 %**        |
| `postprocess`          | 0.00      | 0.0002          | < 0.1 %           |

### Wall-clock breakdown

| Step                | Total (s) | Avg / batch (s) |
|---------------------|-----------|-----------------|
| `read_samples`      | 2.54      | 0.64            |
| `predict_batch_wall`| 318.05    | 79.51           |
| `write_csv`         | 0.01      | 0.004           |

**~77 s per sample** end-to-end on Kaggle GPU.

---

## What the numbers mean

**Generation dominates at 97 % of inference time.** Everything else — image preprocessing, tokenization, disk I/O — is noise by comparison. The bottleneck is autoregressive token generation on the GPU, not data loading or CPU work.

`tokenize_and_transfer` (2.6 %) is the only secondary cost worth watching; padding a batch of mixed-length sequences wastes GPU memory and slows throughput if batch size grows.

`read_samples` (0.64 s/batch) comes entirely from decoding JPEG/PNG from disk; it is currently sequential and un-cached but is irrelevant at the current scale.

---

## Optimization roadmap

### High-impact (targets the 97 % bottleneck)

> `max_new_tokens` is already set to **32** — the minimum viable for coordinate output. Token budget is not the bottleneck; the cost is in the model itself.

| Technique | Expected gain | Notes |
|-----------|--------------|-------|
| **4-bit / 8-bit quantization** (bitsandbytes, GPTQ, AWQ) | 2–4× throughput, ~½ VRAM | Minimal accuracy drop for grounding tasks; enables larger batch sizes on the same GPU |
| **Flash Attention 2** | 1.3–1.8× on long contexts | Drop-in via `attn_implementation="flash_attention_2"` in `from_pretrained`; free win on Kaggle A100/T4 |
| **`torch.compile`** | 5–15 % | Fuse ops; pair with `torch.inference_mode` wrapper |
| **Speculative decoding** | 2–3× | Requires a small draft model; feasible if a smaller HALO2 variant exists |
| **Larger batch size** | Near-linear scaling up to memory limit | Config already sets `batch_size=4`; increase further if VRAM allows after quantization |

### Medium-impact

| Technique | Expected gain | Notes |
|-----------|--------------|-------|
| **KV-cache pinning** | Reduces re-computation on repeated system prompts | Useful if the instruction prefix is shared across samples |
| **Dynamic padding** (sort by length before batching) | Reduces wasted compute in `tokenize_and_transfer` | Relevant at batch_size ≥ 4 |

### Low-impact (already cheap)

- `preprocess_image`, `preprocess_text`, `postprocess` — negligible; no action needed.
- `read_samples` — prefetch with a `DataLoader` worker if batch size > 8.

---

## Recommended next step

Enable 4-bit quantization (bitsandbytes `load_in_4bit=True`) and Flash Attention 2. Both are drop-in config changes with no code rewrite, and together should bring avg latency from ~77 s to ~20–30 s per sample on the same hardware.
