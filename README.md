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
