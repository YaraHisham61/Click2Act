# Plan: Click2Act Research Project Architecture

## Context

The project evaluates multiple GUI agent approaches (OmniParser, AGUVIS, UITARS, etc.) across
multiple benchmarks (OSWorld, MMBench-GUI, ScreenSpot, …). We need a structure that:

- Makes adding a new model or benchmark a 1-file change
- Supports two kinds of model variation:
  - **Param variants** (threshold, prompt template) → config-driven, same class
  - **Algorithmic variants** (different backbone, different grounding strategy) → Python subclass
- Keeps evaluation logic benchmark-agnostic
- Produces reproducible experiment outputs from a single YAML

---

## Design Patterns Used

### 1. Strategy Pattern — `GUIAgent` interface
Each model is a concrete *strategy* implementing the same `predict()` contract.
The evaluator never knows which model it's running.

### 2. Adapter Pattern — `BenchmarkAdapter` interface
Each benchmark expects different input shapes and output formats.
An adapter wraps the benchmark's format on both ends, so the model always
sees / returns a canonical `AgentOutput`.

### 3. Registry Pattern — string → class mapping
YAML configs reference models and benchmarks by name.
A central registry maps those names to classes at runtime, removing all
hard-coded imports from the evaluation pipeline.

### 4. Factory Pattern — `registry.build(name, config)`
Instantiation logic lives in one place. The factory reads the YAML config,
resolves the class from the registry, and injects params. Callers never call
constructors directly.

### 5. Template Method Pattern — base classes with hooks
`GUIAgent` and `BenchmarkAdapter` define the *skeleton* of the algorithm
(load → predict → format_output) and delegate model-specific steps to
subclasses via abstract methods. This is what makes algorithmic variants natural:
subclass overrides only the hook that changed.

---

## Folder Structure

```
src/
├── agents/                     # replaces src/models/ for agent logic
│   ├── base.py                 # GUIAgent ABC + AgentOutput dataclass
│   ├── registry.py             # MODEL_REGISTRY + build() factory
│   ├── omniparser/
│   │   ├── __init__.py
│   │   ├── agent.py            # OmniParserAgent  (base variant)
│   │   └── agent_v2.py         # OmniParserAgentV2 (algorithmic variant)
│   ├── aguvis/
│   │   ├── __init__.py
│   │   └── agent.py
│   └── uitars/
│       ├── __init__.py
│       └── agent.py
├── benchmarks/                 # NEW module
│   ├── base.py                 # BenchmarkAdapter ABC + BenchmarkSample dataclass
│   ├── registry.py             # BENCHMARK_REGISTRY + build() factory
│   ├── osworld.py
│   ├── mmbench_gui.py
│   └── generic.py              # Fallback adapter for new benchmarks
├── pipeline/
│   ├── evaluator.py            # Main evaluation loop (model-agnostic)
│   └── experiment.py           # Loads experiment YAML, wires model+benchmark
├── data/                       # Data loaders (unchanged from README)
├── losses/                     # (unchanged)
└── utils/
    ├── io.py
    ├── metrics.py              # Shared metric helpers (IoU, accuracy, etc.)
    └── viz.py

configs/
├── models/                     # Per-model base configs
│   ├── omniparser.yaml
│   ├── omniparser_v2.yaml      # algorithmic variant — different class
│   ├── aguvis.yaml
│   └── uitars.yaml
└── experiments/                # One file = one run
    ├── 2026-04-11_omniparser_osworld.yaml
    └── 2026-04-11_aguvis_mmbench.yaml

scripts/
└── evaluate.py                 # CLI entry: uv run python -m scripts.evaluate --exp <yaml>

experiments/                    # Output of runs (gitignored)
└── 2026-04-11_omniparser_osworld/
    ├── config.yaml             # copy of the experiment config (reproducibility)
    ├── predictions.jsonl
    └── metrics.json
```

---

## Key Interfaces

### `AgentOutput` — canonical model output format

```python
# src/agents/base.py
@dataclass
class AgentOutput:
    coordinate: tuple[float, float] | None  # normalized (x, y)
    action_type: str | None                 # "click" | "type" | "scroll" | etc 
    text: str | None                        # for "type" actions
    raw: dict                               # model-specific raw output 

### `GUIAgent` — Strategy interface

```python
# src/agents/base.py
class GUIAgent(ABC):
    def __init__(self, config: dict): ...

    @abstractmethod
    def load(self) -> None:
        """Load weights / initialize model."""

    def predict_click(self, screenshot: PIL.Image.Image, instruction: str) -> AgentOutput:
        """Grounding only — return a click coordinate. For example used in ScreenSpot-style benchmarks."""
        raise NotImplementedError

    def predict_action(self, screenshot: PIL.Image.Image, instruction: str) -> AgentOutput:
        """Single-step action — return action_type + coordinate/text. For example used in MMBench-GUI."""
        raise NotImplementedError

    def predict_stateful(self, screenshot: PIL.Image.Image, instruction: str, history: list[AgentOutput],) -> AgentOutput:
        """Multi-step action with prior-step history. Used by OSWorld-style benchmarks."""
        raise NotImplementedError

    # Template method hooks (optional overrides):
    def preprocess(self, screenshot: PIL.Image.Image) -> PIL.Image.Image:
        return screenshot
    def postprocess(self, raw_output) -> AgentOutput: ...
```

Each agent implements only the predict variants its architecture supports.
The benchmark adapter calls the correct variant via `task_mode` (see below).

### `Benchmark` — strategy interface

```python
# src/benchmarks/base.py
class Benchmark(ABC):
    def __init__(self, config: dict|None = None): ...

    def load_samples(self) : ...

    @abstractmethod
    def score(self, predictions: [AgentOutput]) -> [dict]:
        """Return per-sample metrics. """
```

`task_mode` lets the evaluator call the right predict variant without knowing the benchmark.

---

### Registry — wires both tracks together

```python
# src/agents/registry.py
MODEL_REGISTRY: dict[str, type[GUIAgent]] = {
    "omniparser":    OmniParserAgent,
    "omniparser_v2": OmniParserAgentV2,
    "aguvis":        AGUVISAgent,
    "uitars":        UITARSAgent,
}

def build_agent(model_config: dict) -> GUIAgent:
    cls = MODEL_REGISTRY[model_config["class"]]
    agent = cls(model_config.get("params", {}))
    agent.load()
    return agent
```

---

## Experiment Config Schema

```yaml
# configs/experiments/2026-04-11_omniparser_osworld.yaml
experiment:
  name: omniparser_baseline_osworld
  date: 2026-04-11
  notes: "Baseline run, default thresholds"

model:
  config: configs/models/omniparser.yaml   # points to model config

benchmark:
  config: configs/benchmarks/osworld.yaml  # points to benchmark config
  split: test
  max_samples: null                        # null = all

output:
  dir: experiments/2026-04-11_omniparser_osworld/
  save_predictions: true
  save_screenshots: false
```

---

## Evaluation Pipeline

```python
# scripts/evaluate.py
# uv run python -m scripts.evaluate --exp configs/experiments/xxx.yaml

def main(exp_yaml: str):
    exp = load_yaml(exp_yaml)

    agent   = build_agent(load_yaml(exp.model.config))
    bench   = build_benchmark(load_yaml(exp.benchmark.config))
    samples = bench.load_samples(split=exp.benchmark.split)

    scores = []
    for sample in tqdm(samples):
        output = agent.predict(sample.screenshot, sample.instruction)
        score  = bench.score(output, sample)
        scores.append(score)

    metrics = bench.aggregate(scores)
    save_results(exp.output.dir, metrics, scores, exp)
```

The evaluator is **zero knowledge** of which model or benchmark is running.

---

## Adding a New Model (Checklist)

1. Create `src/agents/<modelname>/agent.py` implementing `GUIAgent`
2. Register in `src/agents/registry.py`
3. Add `configs/models/<modelname>.yaml`
4. Done — no changes to evaluator or benchmarks

## Adding a New Benchmark (Checklist)

1. Create `src/benchmarks/<name>.py` implementing `BenchmarkAdapter`
2. Register in `src/benchmarks/registry.py`
3. Add `configs/benchmarks/<name>.yaml`
4. Done

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/agents/base.py` | `GUIAgent` ABC, `AgentOutput` dataclass |
| `src/agents/registry.py` | `MODEL_REGISTRY`, `build_agent()` |
| `src/benchmarks/base.py` | `BenchmarkAdapter` ABC, `BenchmarkSample` |
| `src/benchmarks/registry.py` | `BENCHMARK_REGISTRY`, `build_benchmark()` |
| `src/pipeline/evaluator.py` | evaluation loop |
| `src/pipeline/experiment.py` | YAML loading + wiring |
| `scripts/evaluate.py` | CLI entry point |
| `configs/models/omniparser.yaml` | Example model config |
| `configs/experiments/template.yaml` | Experiment template |

Concrete model/benchmark implementations (`omniparser/agent.py`, `osworld.py`, etc.)
are filled in per-model as work progresses — the interfaces above define their contracts.

---

## Verification

```bash
# Smoke test: run one sample through the whole pipeline
uv run python -m scripts.evaluate \
  --exp configs/experiments/2026-04-11_omniparser_osworld.yaml \
  --max-samples 5

# Check output written
ls experiments/2026-04-11_omniparser_osworld/
# → config.yaml  predictions.jsonl  metrics.json
```
