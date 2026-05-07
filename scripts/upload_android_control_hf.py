# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-01
# Prompt : Download all smolagents/android-control parquet files one-at-a-time,
#          filter rows by episode_id from android_control_splits.json test split,
#          batch into ~500MB parquet files, upload to HuggingFace, delete intermediates.
#          Designed to run on Colab.

import os
import gc
import json
import shutil
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm.auto import tqdm
# ── CONFIG ────────────────────────────────────────────────────────────────────
SRC_REPO       = "smolagents/android-control"
DST_REPO       = "aliaagheis/android-control"    # change to your HF repo
HF_TOKEN       = os.environ.get("HF_TOKEN", "")   # set via Colab Secrets
SPLITS_JSON    = Path("android_control_splits.json")     # path in SRC_REPO data folder
SPLIT          = "test"                            # key inside splits JSON
TARGET_ROWS    = 200
TMP_DIR        = Path("/tmp/android_control")
# ─────────────────────────────────────────────────────────────────────────────


def load_split_ids(splits_json_path: Path, split: str) -> set:
    splits = json.loads(splits_json_path.read_text())
    ids = set(splits[split])
    print(f"Loaded {len(ids)} episode_ids for split='{split}'")
    return ids


def get_parquet_files(repo_id: str, token: str) -> list[str]:
    files = sorted(
        f for f in list_repo_files(repo_id, repo_type="dataset", token=token)
        if f.endswith(".parquet")
    )
    print(f"Found {len(files)} parquet files in {repo_id}")
    return files


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def flush_batch(buffer_dfs: list, batch_idx: int, api: HfApi) -> None:
    out_name = f"test-{batch_idx:05d}.parquet"
    out_path = TMP_DIR / out_name

    combined = pd.concat(buffer_dfs, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    del combined
    gc.collect()

    print(f"  → Batch {batch_idx:03d}: {out_name}  ({mb(out_path):.1f} MB, writing...)")

    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=f"data/{out_name}",
        repo_id=DST_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    out_path.unlink()
    print(f"  → Uploaded and deleted {out_name}")


def main():
    assert HF_TOKEN, "Set HF_TOKEN environment variable before running."
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    test_ids = load_split_ids(SPLITS_JSON, SPLIT)

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=DST_REPO, repo_type="dataset", exist_ok=True)
    print(f"Destination: https://huggingface.co/datasets/{DST_REPO}\n")

    src_files = get_parquet_files(SRC_REPO, HF_TOKEN)

    buffer_dfs: list[pd.DataFrame] = []
    buffer_rows = 0
    batch_idx = 0
    total_kept = 0

    for i, src_path in enumerate(src_files):
        print(f"[{i+1:03d}/{len(src_files)}] {src_path}")

        local_path = Path(
            hf_hub_download(
                repo_id=SRC_REPO,
                filename=src_path,
                repo_type="dataset",
                token=HF_TOKEN,
                local_dir=str(TMP_DIR),
                local_dir_use_symlinks=False,
            )
        )

        df = pd.read_parquet(local_path)
        local_path.unlink()

        filtered = df[df["episode_id"].isin(test_ids)]
        kept = len(filtered)
        total_kept += kept
        print(f"  {kept}/{len(df)} rows kept (source deleted)")

        if kept == 0:
            continue

        buffer_dfs.append(filtered)
        buffer_rows += kept

        is_last = i == len(src_files) - 1
        if buffer_rows > TARGET_ROWS or is_last:
            flush_batch(buffer_dfs, batch_idx, api)
            buffer_dfs.clear()
            buffer_rows = 0
            batch_idx += 1
            gc.collect()
    if buffer_rows > TARGET_ROWS or is_last:
            flush_batch(buffer_dfs, batch_idx, api)
            buffer_dfs.clear()
            buffer_rows = 0
            batch_idx += 1
            gc.collect()

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(f"\nDone. {total_kept} test rows across {batch_idx} files uploaded to {DST_REPO}")


if __name__ == "__main__":
    main()
