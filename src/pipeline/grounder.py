# AI-REFINED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-19
# Prompt : Add batch processing with multi-threading per batch, CSV output via pandas, and __main__ entry point using pathlib
"""
Run An Agent as Grounder (i.e predict x,y) on Dataset/Benchmark and keeps results in file
config:
    - agent         (config path of agent)
    - benchmark     (config path of bench)
    - output_csv    (path to csv file to save results *NOTE* it continue from the last idx in file)
    - batch_size    (default = 1)
    - dataset_size  (optional: number of rows taking after last idx in outpuut_csv)
    - mode          (optional)
        - rerun_failed_ones :   run agent using on failed rows (null values in x,y) using **sample idx**
        - null              :   run agent on samples after last idx in output_csv
NOTE: it very sensitive to output_csv idx column so if any change happen in data some samples will missed and repeat another results (alwayse make sure to check duplicates)              
"""

import gc
import sys
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

        outputs = agent.predict_click_batch([(s.screenshot, s.task) for s in samples])

        rows = [
            {
                'idx'        : batch_start + j,
                'task'       : s.task,
                'coord_x'    : o.coordinate[0] if o.coordinate else None,
                'coord_y'    : o.coordinate[1] if o.coordinate else None,
                'action_type': o.action_type,
                'text'       : o.text,
                'raw_output' : o.raw,
                'annotation' : s.annotation,
            }
            for j, (s, o) in enumerate(zip(samples, outputs))
        ]

        pd.DataFrame(rows).to_csv(
            output_csv, mode='a', header=write_header, index=False
        )
        write_header = False
        gc.collect()
        torch.cuda.empty_cache()

def rerun_failed_ones(config: dict):
    agent_config       = yaml.safe_load(Path(config['agent']).read_text())
    benchmark_config   = yaml.safe_load(Path(config['benchmark']).read_text())

    agent     = build_agent(agent_config)
    benchmark = build_benchmark(benchmark_config)

    batch_size = config.get('batch_size', 1)
    output_csv = Path(config['output_csv'])
    assert output_csv.exists()
    
    results = pd.read_csv(output_csv, index_col=0)
    failed_results = results[results['coord_x'].isna()]
    # ------ generation loop -----
    for batch_start in tqdm(range(0, len(failed_results), batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, len(failed_results))
        
        rows = failed_results.iloc[batch_start:batch_end]
        idxs = list(rows.index)
        samples = [benchmark.get_sample(idx) for idx in idxs]
        # assert they all have same task
        assert all(sample.task == row_task 
                   for sample, row_task in zip(samples, rows['task'])
                ), "Failed to Rerun rows with errors, samples in dataset are not same order"
            
        # generate outputs
        outputs = agent.predict_click_batch([(s.screenshot, s.task) for s in samples])
        # update results df
        cols_to_update = ['coord_x', 'coord_y', 'action_type', 'text', 'raw_output']
        # TODO: Update results
        results.loc[idxs, cols_to_update] = [[
                o.coordinate[0] if o.coordinate else None,
                o.coordinate[1] if o.coordinate else None,
                o.action_type,
                o.text,
                o.raw
            ] for o in outputs
        ]
        gc.collect()
        torch.cuda.empty_cache()
        
    results.to_csv(output_csv,  index=False)

    
if __name__ == '__main__':
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    
    if 'mode' in config and config['mode'] == 'rerun_failed_ones':
        rerun_failed_ones(config)
    else:
        main(config)
