# Refined from grounder.py
"""
Run An Agent as Action Predictpr (i.e predict action) on Dataset/Benchmark and keeps results in file
config:
    - agent         (config path of agent)
    - benchmark     (config path of bench)
    - output_csv    (path to csv file to save results *NOTE* it continue from the last idx in file)
    - batch_size    (default = 1)
    - dataset_size  (optional: number of rows taking after last idx in outpuut_csv)
NOTE: it very sensitive to output_csv idx column so if any change happen in data some samples will missed and repeat another results (alwayse make sure to check duplicates)              
"""

import gc
import sys
import ast
import yaml
import torch
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

    # If I want to batch so I can upload every "dataset_size" row    
    if 'dataset_size' in config:
        size = min(start_idx + config['dataset_size'], benchmark.size)
    else:
        size = benchmark.size

    for batch_start in tqdm(range(start_idx, size, batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, size)
        samples = [benchmark.get_sample(i) for i in range(batch_start, batch_end)]

        outputs = agent.predict_action_batch([(s.screenshot, s.task) for s in samples])

        rows = [
            {
                'idx'        : batch_start + j,
                'task'       : s.task,
                'annotation' : s.annotation,
                # REFINED [old]: o.o.model_dump_json() → [new]: o.model_dump_json()
                'raw_output' : o.model_dump_json(),
            }
            for j, (s, o) in enumerate(zip(samples, outputs))
        ]

        pd.DataFrame(rows).to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == '__main__':
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    
    main(config)
