from huggingface_hub import snapshot_download

repo_id = "rootsautomation/ScreenSpot"
local_directory = "../data/raw/screenspot"

downloaded_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_directory,
)

print(f"\nDownload complete! Files saved in: {downloaded_path}")
