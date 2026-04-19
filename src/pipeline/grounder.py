# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-19
# Prompt : Add batch processing with multi-threading per batch, CSV output via pandas, and __main__ entry point using pathlib
"""
Run An Agent as Grounder (i.e predict x,y) on Dataset/Benchmark and keeps results in file
"""
import sys
import yaml
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm.auto import tqdm

from src.agents import build_agent
from src.benchmarks import build_benchmark


def main(config: dict):
    agent_config       = yaml.safe_load(Path(config['agent']).read_text())
    benchmark_config   = yaml.safe_load(Path(config['benchmark']).read_text())

    agent     = build_agent(agent_config)
    benchmark = build_benchmark(benchmark_config)

    batch_size = config.get('batch_size', 1)
    output_csv = Path(config['output_csv'])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # REFINED [old]: always start from 0 → [new]: resume from last written idx
    start_idx = 0
    write_header = True
    if output_csv.exists():
        done = pd.read_csv(output_csv)
        done = done[done['idx'] != 'idx']  # drop duplicate header rows
        if not done.empty:
            start_idx = int(done['idx'].astype(int).max()) + 1
            write_header = False

    logger.debug("Load All Images at once")
    all_samples = [benchmark.get_sample(i) for i in range(start_idx, benchmark.size)]
    logger.debug(f"Start generating with batchs {batch_size}")
    
    for batch_start in tqdm(range(start_idx, benchmark.size, batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, benchmark.size)
        samples = all_samples[batch_start:batch_end]

        outputs = agent.predict_click_batch([(s.screenshot, s.task) for s in samples])

        rows = [
            {
                'idx'        : batch_start + j,
                'task'       : s.task,
                'coord_x'    : o.coordinate[0] if o.coordinate else None,
                'coord_y'    : o.coordinate[1] if o.coordinate else None,
                'action_type': o.action_type,
                'text'       : o.text,
                'annotation' : s.annotation,
            }
            for j, (s, o) in enumerate(zip(samples, outputs))
        ]

        pd.DataFrame(rows).to_csv(
            output_csv, mode='a', header=write_header, index=False
        )
        write_header = False


if __name__ == '__main__':
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    main(config)
