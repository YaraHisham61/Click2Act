# Research Roadmap: Click2Act

## Context

You've completed environment setup: 3 models downloaded (UI-TARS 7B, AGUVIS 7B, OmniParser), 3 conda envs installed, datasets in `data/raw/`. The project goal is to compare autonomous GUI agents across design choices (vision-only vs. hybrid). No experiments have run yet. This plan answers: **where do you start?**

---

## High-Level Research Strategy

The project is an **evaluation study** with potential for a small **contribution** (not a new model). The arc is:

```
Sanity Check → Grounding Benchmark → Task Benchmark → Analysis → Contribution
```

Start narrow (single task type, offline, no VM), expand to full pipelines only after baselines are confirmed reproducible.

---

## Phase 0 — Smoke Tests (this week)

**Goal:** Confirm each model runs inference end-to-end on one sample, with known-correct output.

For each model, write a minimal script in `scripts/` that:
1. Loads the model
2. Runs one screenshot + instruction → predicted action
3. Prints raw output

### Practical order (easiest → hardest)

| Model | Conda Env | Entry Point | Input Format |
|-------|-----------|-------------|--------------|
| OmniParser | omniparser (py3.12) | `external/OmniParser/` | image → parsed elements |
| AGUVIS | aguvis (py3.10) | `external/aguvis/src/aguvis/serve/cli.py` | image + instruction → pyautogui action |
| UI-TARS | uitars env | `ui_tars` package (installed) | image + instruction → action |

**Deliverable:** `scripts/smoke_test_<model>.py` for each model. Once all three pass → Phase 1.

---

## Phase 1 — GUI Grounding Benchmark (ScreenSpot)

**Why start here:** Grounding is the simplest task type — given screenshot + element description → predict (x,y) coordinate. No execution environment (no Docker/VM). Lets you compare all 3 models on equal footing quickly.

**Dataset to use:** ScreenSpot (you have ScreenSpot-Pro leaderboard screenshots in `docs/assets/`). Need to download the actual dataset — ~150MB, no VM required.

**Metric:** Localization accuracy = fraction of predictions within element bounding box.

### What to run
```
scripts/eval_grounding_omniparser.py
scripts/eval_grounding_aguvis.py
scripts/eval_grounding_uitars.py
```

Each script: loads N samples from ScreenSpot → runs model → computes accuracy per platform (web/desktop/mobile).

**Deliverable:** Table: model × platform → accuracy. This is your first result.

---

## Phase 2 — Offline Task Benchmark (Mind2Web)

**Why Mind2Web:** You already have it (`data/raw/mind2web/`). It's offline (no live browser), provides screenshots + action sequences. Well-established baseline numbers to compare against.

**Metric:** Element Accuracy (EA) — does the model predict the correct element per step?

**What to run:** Use `data/raw/aguvis-stage-2/mind2web-l*.json` (already formatted, L1/L2/L3 difficulty levels) — this avoids re-processing the raw Mind2Web format.

**Deliverable:** Table: model × difficulty level → EA.

---

## Phase 3 — Live Task Benchmark (OSWorld, subset)

**Why last:** Requires Docker + virtual desktop (KVM). More setup, slower, higher variance. Start only after Phases 1–2 give you confident baselines.

**Recommended subset:** 50–100 tasks from OSWorld across 3–4 app categories (e.g., Chrome, Files, VS Code, LibreOffice). Full OSWorld is 369 tasks — expensive to run.

**Entry point:** `external/OSWorld/lib_run_single.py`

---

## Phase 4 — Analysis & Contribution

After baselines, look for one of these angles for a research contribution:

### Option A: Error taxonomy
Manually label ~50 failures per model. Categorize: wrong element, correct element wrong action, hallucinated element, OCR failure, resolution-dependent. Ask: *do different models fail on different categories?*

### Option B: OmniParser as perception module for AGUVIS/UI-TARS
OmniParser decomposes UI understanding into: detect icons → OCR text → label elements. Hypothesis: replacing each model's native perception with OmniParser's parsed representation improves accuracy on icon-heavy UIs. Test on ScreenSpot icon subset.

### Option C: Resolution ablation
AGUVIS was trained at 720P. Run AGUVIS at 360P / 720P / 1080P on the same ScreenSpot samples. Does resolution affect performance differently across web/desktop/mobile?

### Option D: Cross-domain transfer
Models trained on web (Mind2Web) — how much do they degrade on desktop (OSWorld)? Is the gap larger for AGUVIS vs. UI-TARS?

**Recommendation:** Start with Option A (error taxonomy) — it's parallelizable with Phase 3 and doesn't require new code. Options B or D are strongest for a thesis contribution.

---

## Practical First Steps (Today / Tomorrow)

1. **Write `scripts/smoke_test_omniparser.py`** — easiest model, pure inference, no action parsing needed.
2. **Download ScreenSpot dataset** — check HuggingFace (`rootsautomation/ScreenSpot`), ~150MB.
3. **Sketch eval loop structure** — all eval scripts share: `load_dataset() → for sample in dataset: predict() → compute_metric()`. Write this as `src/eval/base_evaluator.py` once, reuse.

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/smoke_test_omniparser.py` | Verify OmniParser runs |
| `scripts/smoke_test_aguvis.py` | Verify AGUVIS runs |
| `scripts/smoke_test_uitars.py` | Verify UI-TARS runs |
| `scripts/download_dataset_screenspot.py` | Download ScreenSpot |
| `src/eval/base_evaluator.py` | Shared eval loop |
| `scripts/eval_grounding_<model>.py` | Grounding evals (×3) |
| `notebooks/01_explore_datasets.ipynb` | EDA on Mind2Web / ScreenSpot |

---

## Verification

After Phase 1:
- All 3 models produce localization accuracy numbers on ScreenSpot.
- Numbers are within ±5% of published leaderboard values (sanity check).
- Scripts runnable with: `conda activate <env> && uv run python -m scripts.eval_grounding_omniparser`

After Phase 2:
- Mind2Web EA scores comparable to AGUVIS paper (Table 2 of `references/solution-AGUVIS.pdf`).
