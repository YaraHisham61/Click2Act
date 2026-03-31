# Click2Act
Evaluating Autonomous Agents for GUI Interaction


SAM Prompt Analysis &amp; Robustness Evaluation

## Folder Structure
```text
Click2Act/
├── data/                   # Data is kept separate from code
│   ├── raw/                # Immutable original datasets
│   ├── processed/          # Preprocessed data (e.g., resized, normalized)
│   └── external/           # Third-party data or metadata
├── src/                    # Main source code (as a package)
│   ├── __init__.py
│   ├── data/               # Data loaders and augmentation logic
│   ├── models/             # Model architectures (backbones, heads)
│   ├── losses/             # Custom loss functions
│   ├── utils/              # Helper functions (logging, visualization)
│   └── pipeline/           # Training and inference loops
├── configs/                # Experiment configurations (YAML/JSON)
├── scripts/                # Entry points for CLI execution
│   ├── train.py
│   ├── evaluate.py
│   └── preprocess.py
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── experiments/            # Output of runs
│   └── 2024-03-26_exp1/    # Logs, checkpoints, and visualizations per run
├── docs/                   # Documentation and research notes
├── references/             # Papers, manual, and explanatory material
├── tests/                  # Unit tests for core logic
├── .gitignore              # Pre-configured to ignore data/ and experiments/
├── requirements.txt        # Python dependencies
└── README.md               # Project overview and setup instructions

```

## Installation

This project uses **uv** on top of a **conda** base environment.

Conda should provide heavy libraries
- `torch`

and uv manages the rest inside `.venv`.
**activate your conda env** (Python 3.12, with torch + transformers already installed)
```bash
conda activate <your-env>
```

**create `.venv` using conda's Python, inheriting its site-packages**
```bash
pip install uv

uv venv --python $(which python) --system-site-packages
```

**install dependencies**
```bash
uv sync
```

**activate**
```bash
# linux / macOS
source .venv/bin/activate
# windows
.venv\Scripts\activate
```


## Branching & Commit Rules

### Branch Naming


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

---

## AI Usage Disclaimer

> [!IMPORTANT]
> **AI Transparency**: every file, line using AI will be mentioned, you can look on CLAUDE.md and GEMINI.md to see strict rules for AI transparency

This project uses AI assistants (claude and gemini) under strict transparency and containment rules.

### Transparency

Every file or line produced or modified by an AI is marked with header:

```
<!-- AI-GENERATED
     Model   : <model name>
     Date    : YYYY-MM-DD
     Prompt  : <prompt summary>
-->
```

Every inline edit is tracked with a `REFINED` comment directly above the changed line. 


> See `CLAUDE.md` and `GEMINI.md` for the full rules.

### AI-generated files location

| Tool   | Output folder   |
|--------|-----------------|
| claude | `docs/claude/`  |
| gemini | `docs/gemini/`  |

`docs/` (root) is reserved for human-authored documentation only.

### Context isolation

AI tools are prohibited from reading their own generated output (`docs/claude/`, `docs/gemini/`). 

These folders will contain noisy, outdated, and incorrect notes. Also recent research shows that AI reading its own rules or notes decrease its productivity and efficeny. 
