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
        - rerun_failed_ones :   run agent using on failed rows (null values in x,y) using **annotation** dict
        - null              :   run agent on samples after last idx in output_csv
NOTE: it very sensitive to output_csv idx column so if any change happen in data some samples will missed and repeat another results (alwayse make sure to check duplicates)              
"""

import gc
import sys
import ast
import time
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
    save_every_n = int(config.get('save_every_n', 300))
    sample_timeout_s = config.get('sample_timeout_s', None)
    skip_too_slow = config.get('skip_too_slow', True)
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

    ok_count = 0
    failed_count = 0
    timeout_count = 0
    pending_rows = []

    def flush_rows(force: bool = False) -> None:
        nonlocal write_header, pending_rows
        if not pending_rows:
            return
        if len(pending_rows) >= save_every_n or force:
            pd.DataFrame(pending_rows).to_csv(
                output_csv, mode='a', header=write_header, index=False
            )
            write_header = False
            pending_rows = []

    for batch_start in tqdm(range(start_idx, size, batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, size)
        samples = [benchmark.get_sample(i) for i in range(batch_start, batch_end)]

        rows = []
        for j, s in enumerate(samples):
            idx = batch_start + j
            t0 = time.perf_counter()
            status = 'ok'
            failed_reason = None

            try:
                o = agent.predict_click(s.screenshot, s.task)
                elapsed_s = time.perf_counter() - t0

                if sample_timeout_s is not None and elapsed_s > float(sample_timeout_s):
                    if skip_too_slow:
                        status = 'timeout'
                        failed_reason = f'elapsed_s={elapsed_s:.3f} > sample_timeout_s={float(sample_timeout_s):.3f}'
                        coord_x, coord_y = None, None
                        timeout_count += 1
                    else:
                        coord_x = o.coordinate[0] if o.coordinate else None
                        coord_y = o.coordinate[1] if o.coordinate else None
                        ok_count += 1 if o.coordinate else 0
                        failed_count += 0 if o.coordinate else 1
                else:
                    coord_x = o.coordinate[0] if o.coordinate else None
                    coord_y = o.coordinate[1] if o.coordinate else None
                    if o.coordinate is None:
                        status = 'failed'
                        failed_reason = 'coordinate_missing'
                        failed_count += 1
                    else:
                        ok_count += 1

                row = {
                    'idx'        : idx,
                    'task'       : s.task,
                    'coord_x'    : coord_x,
                    'coord_y'    : coord_y,
                    'action_type': o.action_type,
                    'text'       : o.text,
                    'raw_output' : o.raw,
                    'annotation' : s.annotation,
                    'status'     : status,
                    'failed_reason': failed_reason,
                    'latency_s'  : elapsed_s,
                }
            except Exception as err:
                elapsed_s = time.perf_counter() - t0
                failed_count += 1
                row = {
                    'idx'        : idx,
                    'task'       : s.task,
                    'coord_x'    : None,
                    'coord_y'    : None,
                    'action_type': None,
                    'text'       : None,
                    'raw_output' : None,
                    'annotation' : s.annotation,
                    'status'     : 'failed',
                    'failed_reason': repr(err),
                    'latency_s'  : elapsed_s,
                }

            rows.append(row)

        pending_rows.extend(rows)
        flush_rows(force=False)
        gc.collect()
        torch.cuda.empty_cache()

    flush_rows(force=True)

    logger.info(
        "Grounder run complete | total={} ok={} failed={} timeout={} output_csv={}",
        max(0, size - start_idx),
        ok_count,
        failed_count,
        timeout_count,
        output_csv,
    )

def rerun_failed_ones(config: dict):
    agent_config       = yaml.safe_load(Path(config['agent']).read_text())
    benchmark_config   = yaml.safe_load(Path(config['benchmark']).read_text())

    agent     = build_agent(agent_config)
    benchmark = build_benchmark(benchmark_config)

    batch_size = config.get('batch_size', 1)
    output_csv = Path(config['output_csv'])
    assert output_csv.exists()
    
    results = pd.read_csv(output_csv, index_col=0)
    results['annotation'] = results['annotation'].apply(ast.literal_eval)
    failed_results = results[results['coord_x'].isna()]
    # ------ generation loop -----
    for batch_start in tqdm(range(0, len(failed_results), batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, len(failed_results))
        
        rows = failed_results.iloc[batch_start:batch_end]
        idxs = list(rows.index)
        samples = [benchmark.get_sample_from_annotation(ann) for ann in rows['annotation']]
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
        
    results.to_csv(output_csv,  index=True)

    
if __name__ == '__main__':
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    
    if 'mode' in config and config['mode'] == 'rerun_failed_ones':
        rerun_failed_ones(config)
    else:
        main(config)
