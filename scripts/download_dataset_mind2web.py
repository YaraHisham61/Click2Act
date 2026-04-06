# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-04-05
# Prompt : Create script to download subset of Mind2Web dataset: train_0/2/4/6.json, test.zip, scores_all_data.pkl, readme.md
from huggingface_hub import snapshot_download

repo_id = "osunlp/Mind2Web"
local_directory = "../data/raw/mind2web"

target_files = [
    "data/train/train_0.json",
    "data/train/train_2.json",
    "data/train/train_4.json",
    "data/train/train_6.json",
    "test.zip",
    "scores_all_data.pkl",
    "readme.md",
]

print("Starting download for Mind2Web (partial train + test)...")

downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=target_files,
    local_dir=local_directory,
)

print(f"\nDownload complete! Files saved in: {downloaded_path}")
