# AI-GENERATED
# Date   : 2026-04-05
from huggingface_hub import snapshot_download

repo_id = "xlangai/aguvis-stage2"
local_directory = "../data/raw/aguvis-stage-2"

# We use the wildcard (*) to grab both the .json and .zip files for each dataset
target_files = [
    "aitw*",       
    "mind2web*",           
    "miniwob*",         
]

downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=target_files,
    local_dir=local_directory
)
