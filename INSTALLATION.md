# Create Base Conda Environment
most external packages required "python=3.10" or "python=3.12" with torch, transformers libraries

For easiness, create a base envs first
```bash
conda create -n base-310 python=3.10 -y 
conda create -n base-312 python=3.12 -y
```
and inside both base envs install `torch` `torchaudio` `torchvision` `transformers`

# Download Datasets & Models
there are three scripts you can run to download datasets
```bash
uv run python -m scripts.download_dataset_aguvis_stage1

uv run python -m scripts.download_dataset_aguvis_stage2

uv run python -m scripts.download_dataset_mind2web
```
there are one script to download three models (you can comment and remove one)
```bash
sh scripts/download_models.sh
```

# Install Packages
## 1. OSWorld
```bash
conda create --name osworld --clone base-310
conda activate osworld

cd external
git clone https://github.com/xlang-ai/OSWorld.git
cd OSWorld

pip install -r requirements.txt
```
After that, we need to run virtual environment of the agent. We used docker on wsl but if yours doesn't support KVM, you follow other ways inside the repo.

## 2. Aguvis
```bash
conda create --name aguvis --clone base-310
conda activate aguvis

cd external
git clone https://github.com/xlang-ai/aguvis
cd aguvis

pip install -r requirements.txt
```

## 3. OmniParser
```bash
conda create --name omniparser --clone base-312
conda activate omniparser

cd external
git clone https://github.com/microsoft/omniparser
cd OmniParser

pip install -r requirements.txt

# paddleocr 3.x dropped legacy parameters used by OmniParser; pin to 2.x
pip install "paddleocr==2.8.0" "paddlepaddle==2.6.2" "google" "google-genai"
```

### Fix: `ImportError: cannot import name 'Sentinel' from 'typing_extensions'`

This error occurs when using `google-genai` or `pydantic-core` because conda installs
`typing_extensions` 4.11.0 and leaves its `.py` file on disk. `pip install --upgrade`
only updates the dist-info metadata without overwriting the stale conda-placed file,
so the running code is still 4.11.0 (which has no public `Sentinel`).

Fix with a force-reinstall:
```bash
pip install --force-reinstall "typing_extensions==4.15.0"
```