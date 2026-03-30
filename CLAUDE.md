# CLAUDE.md — Instructions for Claude Code in Click2Act

This file governs how Claude Code must behave in this research project.
It is authoritative. Follow every rule here without exception.

---

## 1. AI Transparency (Non-Negotiable)

Every file Claude **creates** must begin with a header block:

```
<!-- AI-GENERATED
     Model   : Claude Sonnet 4.6
     Date    : YYYY-MM-DD
     Prompt  : <exact prompt or summary that triggered creation of this file>
-->
```

For **Python / scripts**, use the equivalent comment block:

```python
# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : YYYY-MM-DD
# Prompt : <exact prompt or summary>
```

Never omit this header. If a file grows from an earlier AI-generated file, keep the original header and append a `# Refined:` line (see Rule 2).

---

## 2. Inline Refinement Tracking

When Claude edits an **existing** line (spelling, logic, wording, structure), mark the change inline so the diff is human-readable inside the file itself:

**Markdown / text files:**
```
<!-- REFINED [old]: "Probem Definiation" → [new]: "Problem Definition" -->
```

Place the comment on the line **immediately above** the changed line.

**Python / code files:**
```python
# REFINED [old]: foo(x, y) → [new]: bar(x, y)
```

Place the comment on the line immediately above the changed line.

Rules:
- One `REFINED` comment per changed line.
- Do not stack multiple changes in one comment — split them.
- Do not add `REFINED` comments for trivial whitespace-only fixes.

---

## 3. Research & Markdown Files Location

All Claude-authored or Claude-assisted markdown files (notes, summaries, literature reviews, experiment logs, analyses) **must** live under:

```
docs/claude/
```

Naming convention: `YYYY-MM-DD_<short-slug>.md`
Example: `docs/claude/2026-03-30_omniparser-summary.md`

Every such file must include the full prompt at the top inside the AI-GENERATED header.

The folder `docs/` is for **human-authored** documentation.
The folder `docs/claude/` is for **AI-assisted** research notes.
Do not mix them.

---

## 4. Fixed Plot Color Palette

Use **exactly** this palette for every plot, figure, and visualization in this project. Never deviate.

```python
# Click2Act canonical palette
C2A_PALETTE = {
    "primary"   : "#2E86AB",   # Steel Blue    — main model / baseline
    "secondary" : "#A23B72",   # Raspberry     — comparison model
    "tertiary"  : "#F18F01",   # Amber         — third model / ablation
    "neutral"   : "#6C757D",   # Slate Gray    — reference lines, grids
    "success"   : "#3BB273",   # Emerald       — positive results
    "warning"   : "#E84855",   # Crimson       — failure / error cases
    "bg"        : "#F8F9FA",   # Off-White     — figure background
    "text"      : "#212529",   # Near-Black    — labels, titles
}

# Ordered list for multi-series plots
C2A_ORDER = [
    C2A_PALETTE["primary"],
    C2A_PALETTE["secondary"],
    C2A_PALETTE["tertiary"],
    C2A_PALETTE["success"],
    C2A_PALETTE["warning"],
    C2A_PALETTE["neutral"],
]
```

Apply this palette via `matplotlib`, `seaborn`, or any plotting library.
Always set figure background to `bg` and all text to `text`.
This palette is colorblind-safe under deuteranopia simulation.

---

## 5. Folder Structure Guardian

The canonical folder structure for this project is defined in `README.md`. Claude must:

1. **Warn immediately** if a file is being created outside its correct folder.
2. **State the correct path** alongside the warning.
3. **Ask for confirmation** before proceeding to the wrong location.

### Canonical locations

| What                          | Where                          |
|-------------------------------|--------------------------------|
| Raw datasets                  | `data/raw/`                    |
| Preprocessed data             | `data/processed/`              |
| External / third-party data   | `data/external/`               |
| Source package                | `src/`                         |
| Data loaders                  | `src/data/`                    |
| Model architectures           | `src/models/`                  |
| Loss functions                | `src/losses/`                  |
| Utilities / helpers           | `src/utils/`                   |
| Training / inference loops    | `src/pipeline/`                |
| YAML/JSON configs             | `configs/`                     |
| CLI entry points              | `scripts/`                     |
| Jupyter notebooks (EDA)       | `notebooks/`                   |
| Experiment outputs / runs     | `experiments/<date>_<name>/`   |
| Human-authored docs           | `docs/`                        |
| Claude-assisted research docs | `docs/claude/`                 |
| Papers and references         | `references/`                  |
| Unit tests                    | `tests/`                       |

Warning format to use:
```
⚠️  FOLDER STRUCTURE WARNING
    You are placing `<filename>` in `<proposed path>`.
    The correct location is: `<correct path>`
    Fix: mv <proposed path>/<filename> <correct path>/<filename>
    Proceed anyway? (yes/no)
```

---

## 6. Voice-Preserving Editing (Spelling / Grammar / Markdown)

When asked to fix spelling, grammar, or markdown structure:

### What Claude MUST do
- Fix spelling errors.
- Fix grammar errors (subject-verb agreement, punctuation, article usage).
- Fix markdown structure: heading levels, table alignment, numbered lists, code block fences.
- Keep every word the user wrote.

### What Claude MUST NOT do
- Rephrase sentences.
- Replace words with "better" synonyms.
- Add or remove content.
- Change the user's tone, register, or style.

### After every edit, Claude MUST append a "Word Suggestions" block

Place this block at the **end of the response** (never inside the file):

```
---
### Word Suggestions (optional — your choice)
| Location | Your word | Suggested alternative | Reason |
|----------|-----------|-----------------------|--------|
| §2, line 4 | "compare" | "benchmark" | more precise in evaluation context |
```

The user decides whether to accept any suggestion. Claude never applies suggestions automatically.

### The command to invoke this mode

When the user says `/fix-voice` or "fix spelling/grammar keep my voice", apply Rule 6.

---

## 8. Branching & Commit Rules

### Branch Naming

```
<type>-<short-slug>
```


| Type        | When to use                                      | Example                          |
|-------------|--------------------------------------------------|----------------------------------|
| `feat-`     | New feature or experiment                        | `feat-omniparser-eval`           |
| `fix-`      | Bug fix                                          | `fix-bbox-offset-error`          |
| `data-`     | Data processing or dataset changes               | `data-osworld-preprocessing`     |
| `exp-`      | Exploratory / throwaway experiment               | `exp-attention-viz`              |
| `docs-`     | Documentation or research notes only             | `docs-literature-review`         |
| `refactor-` | Code restructure, no behavior change             | `refactor-pipeline-cleanup`      |

- `master` is the stable, always-runnable branch. Never commit broken code directly to `master`.
- Merge to `master` only after stable delivery. 
- Merge to `dev` when the experiment or feature is complete and reproducible.

### Commit Message Format

```
<type>(<scope>): <short imperative summary>

[optional body — what and why, not how]
[optional: Refs: <paper>, <issue>]
```

Types: `feat`, `fix`, `data`, `exp`, `docs`, `refactor`, `test`, `chore`

Examples:
```
feat(eval): add step-success-rate metric for OSWorld

data(osworld): filter tasks with missing screenshots

exp(omniparser): compare bbox precision at 0.5 vs 0.7 threshold

docs(proposal): fix spelling in problem definition
```

Rules:
- Subject line ≤ 50 characters, imperative mood ("add", not "added").
- No period at the end of the subject line.
- Body explains *why*, not *what* — the diff shows *what*.
- Claude must propose a commit message whenever it makes a meaningful change, but never commit without explicit user approval.

---

## 7. General Research Project Behaviors

- **Cite sources**: whenever referencing a paper, use `[Author et al., Year]` format and link to the PDF in `references/` if available.
- **No speculation without flagging**: if Claude is uncertain, prefix the statement with `> [Uncertain]`.
- **Short responses by default**: research context means dense, precise output — no filler.
- **Reproducibility**: every script must be runnable end-to-end with `uv run python -m scripts.<name>`.
