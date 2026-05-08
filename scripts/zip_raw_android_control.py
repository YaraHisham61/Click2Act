# AI-GENERATED
# Model  : Claude Sonnet 4.6
# Date   : 2026-05-01
# Prompt : Download full aliaagheis/android-control repo, zip data folder

import os
import tarfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

# ── CONFIG ────────────────────────────────────────────────────────────────────
SRC_REPO = "aliaagheis/android-control"
DST_REPO = SRC_REPO
HF_TOKEN = os.environ.get("HF_TOKEN", "")
TMP_DIR  = Path("/tmp/android_control")
# ─────────────────────────────────────────────────────────────────────────────

def main():
    assert HF_TOKEN, "Set HF_TOKEN environment variable before running."

    # ── Download entire repo ──────────────────────────────────────────────────
    print(f"Downloading {SRC_REPO} ...")
    snapshot_download(
        repo_id=SRC_REPO,
        repo_type="dataset",
        local_dir=str(TMP_DIR),
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

    # ── Zip data/ folder ─────────────────────────────────────────────────────
    data_dir = TMP_DIR / "data"
    out_zip  = TMP_DIR / "android_control.tar.gz"
    print(f"\nCompressing {data_dir} ...")
    with tarfile.open(out_zip, "w:gz") as tar:
        tar.add(data_dir, arcname="data")
    shutil.rmtree(data_dir)
    print(f"Archive: {out_zip.stat().st_size / 1e6:.1f} MB")

    # ── Upload ────────────────────────────────────────────────────────────────
    print(f"\nUploading to {DST_REPO} ...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(out_zip),
        path_in_repo="data/android_control.tar.gz",
        repo_id=DST_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("Done.")

if __name__ == "__main__":
    main()
