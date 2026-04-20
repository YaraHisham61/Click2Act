# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-20
# Prompt : copy grounder.py and create profiling_grounder.py where each step is evaluated
#          (generation and writing / inside it create new class Halo2Eval where you override
#          predict_batch to measure time of reading images / preprocessing image /
#          preprocessing texts / actual generation text / postprocessing)
"""
Profiling wrapper around grounder.py — measures time for each step in predict_click_batch.
"""
import sys
import time
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm
from PIL.Image import Image

from src.agents import build_agent
from src.agents.halo2 import HALO2Agent
from src.agents.base import AgentOutput
from src.benchmarks import build_benchmark


class StepTimer:
    """Accumulates wall-clock seconds per named step."""
    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def record(self, step: str, elapsed: float):
        self.totals[step] = self.totals.get(step, 0.0) + elapsed
        self.counts[step] = self.counts.get(step, 0) + 1

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "step": step,
                "total_s": round(self.totals[step], 4),
                "calls": self.counts[step],
                "avg_s": round(self.totals[step] / self.counts[step], 4),
            }
            for step in self.totals
        ]
        return pd.DataFrame(rows)


class Halo2Eval(HALO2Agent):
    """HALO2Agent with per-step timing instrumentation in predict_click_batch."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.timer = StepTimer()

    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        texts, all_images = [], []

        # ── step 1: preprocess images ─────────────────────────────────────────
        t0 = time.perf_counter()
        for screenshot, _ in inputs:
            all_images.append(self.preprocess(screenshot))
        self.timer.record("preprocess_image", time.perf_counter() - t0)

        # ── step 2: preprocess texts (chat template) ──────────────────────────
        t0 = time.perf_counter()
        for (_, task), screenshot_processed in zip(inputs, all_images):
            messages = self._get_grounding_chat_messages(screenshot_processed, task)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, thinking=False
            )
            texts.append(text)
        self.timer.record("preprocess_text", time.perf_counter() - t0)

        # ── step 3: tokenize + move to device ────────────────────────────────
        t0 = time.perf_counter()
        batch_inputs = self.processor(text=texts, images=all_images, padding=True, return_tensors="pt")
        batch_inputs = batch_inputs.to(self.model.device)
        self.timer.record("tokenize_and_transfer", time.perf_counter() - t0)

        # ── step 4: actual generation ─────────────────────────────────────────
        t0 = time.perf_counter()
        generated_ids = self.model.generate(**batch_inputs, max_new_tokens=self.config["max_new_tokens"])
        generated_ids_trimmed = [out[len(inp):] for inp, out in zip(batch_inputs.input_ids, generated_ids)]
        output_texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        self.timer.record("generation", time.perf_counter() - t0)

        # ── step 5: postprocess ───────────────────────────────────────────────
        t0 = time.perf_counter()
        results = [self.postprocess_grounding(t.strip()) for t in output_texts]
        self.timer.record("postprocess", time.perf_counter() - t0)

        return results

    def print_timing_summary(self):
        df = self.timer.summary()
        logger.info("\n=== Halo2Eval Timing Summary ===\n" + df.to_string(index=False))


def main(config: dict):
    agent_config     = yaml.safe_load(Path(config["agent"]).read_text())
    benchmark_config = yaml.safe_load(Path(config["benchmark"]).read_text())

    # force use of Halo2Eval regardless of what the config says
    agent_config["class"] = "Halo2Eval"
    agent = Halo2Eval(agent_config)
    agent.load()

    benchmark = build_benchmark(benchmark_config)

    batch_size  = config.get("batch_size", 1)
    total_size  = config.get("total_size", benchmark.size)
    output_csv  = Path(config["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    start_idx  = 0
    write_header = True
    if output_csv.exists():
        done = pd.read_csv(output_csv)
        done = done[done["idx"] != "idx"]
        if not done.empty:
            start_idx    = int(done["idx"].astype(int).max()) + 1
            write_header = False

    # ── per-batch timing (includes I/O) ──────────────────────────────────────
    batch_timer = StepTimer()

    for batch_start in tqdm(range(start_idx, min(total_size, benchmark.size), batch_size), desc="batches"):
        batch_end = min(batch_start + batch_size, benchmark.size)

        # ── read images from disk ─────────────────────────────────────────────
        t0 = time.perf_counter()
        samples = [benchmark.get_sample(i) for i in range(batch_start, batch_end)]
        batch_timer.record("read_samples", time.perf_counter() - t0)

        # ── predict (internally timed by Halo2Eval) ───────────────────────────
        t0 = time.perf_counter()
        outputs = agent.predict_click_batch([(s.screenshot, s.task) for s in samples])
        batch_timer.record("predict_batch_wall", time.perf_counter() - t0)

        # ── write CSV ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        rows = [
            {
                "idx"        : batch_start + j,
                "task"       : s.task,
                "coord_x"   : o.coordinate[0] if o.coordinate else None,
                "coord_y"   : o.coordinate[1] if o.coordinate else None,
                "action_type": o.action_type,
                "text"       : o.text,
                "raw_output" : o.raw,
                "annotation" : s.annotation,
            }
            for j, (s, o) in enumerate(zip(samples, outputs))
        ]
        pd.DataFrame(rows).to_csv(output_csv, mode="a", header=write_header, index=False)
        write_header = False
        batch_timer.record("write_csv", time.perf_counter() - t0)

    # ── final timing report ───────────────────────────────────────────────────
    agent.print_timing_summary()
    logger.info("\n=== Batch-Level Timing (wall clock) ===\n" + batch_timer.summary().to_string(index=False))


if __name__ == "__main__":
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    main(config)
