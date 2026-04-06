# AI-GENERATED
# Date   : 2026-04-05
from huggingface_hub import snapshot_download

repo_id = "xlangai/aguvis-stage1"
local_directory = "../data/raw/aguvis-stage-1"

target_files = [
    "ricoig16k.*",         # Rico Interaction Graph
    "ricosca.*",           # Rico Semantic Component Analysis
    "ui_refexp.*",         # UI Referring Expressions
    "widget_captioning.*"  # Widget Captioning
]

print("Starting download for Rico, UI RefExp, and Widget Captioning...")

downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=target_files,
    local_dir=local_directory
)

print(f"\nDownload complete! Your selected datasets are saved in: {downloaded_path}")