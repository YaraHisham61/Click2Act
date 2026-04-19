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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from src.agents import build_agent
from src.benchmarks import build_benchmark


def _run_sample(args):
    i, agent, sample = args
    output = agent.predict_click(sample.screenshot, sample.task)
    return {
        'idx'        : i,
        'task'       : sample.task,
        'coord_x'    : output.coordinate[0] if output.coordinate else None,
        'coord_y'    : output.coordinate[1] if output.coordinate else None,
        'action_type': output.action_type,
        'text'       : output.text,
        'annotation' : sample.annotation,
    }


def main(config: dict):
    agent_config       = yaml.safe_load(Path(config['agent']).read_text())
    benchmark_config   = yaml.safe_load(Path(config['benchmark']).read_text())
    
    agent     = build_agent(agent_config)
    benchmark = build_benchmark(benchmark_config)

    batch_size = config.get('batch_size', 1)
    output_csv = Path(config['output_csv'])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = True

    for batch_start in tqdm(range(0, benchmark.size, batch_size), desc='batches'):
        batch_end = min(batch_start + batch_size, benchmark.size)
        args = [
            (i, agent, benchmark.get_sample(i))
            for i in range(batch_start, batch_end)
        ]

        rows = []
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = {pool.submit(_run_sample, a): a[0] for a in args}
            for future in as_completed(futures):
                rows.append(future.result())

        rows.sort(key=lambda r: r['idx'])
        pd.DataFrame(rows).to_csv(
            output_csv, mode='a', header=write_header, index=False
        )
        write_header = False


if __name__ == '__main__':
    config_path = Path(sys.argv[1])
    config = yaml.safe_load(config_path.read_text())
    main(config)
