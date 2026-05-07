from src.benchmarks.base import Benchmark, BenchmarkSample
from src.agents.base import AgentOutput
from src.constants import DATA_PATH
from src.utils import extract_tar

import math
import pandas as pd
from PIL import Image
from io import BytesIO
import base64, textwrap
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download


class AndroidControlBenchmark(Benchmark):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)
        self.config.setdefault("benchmark_path", str(DATA_PATH / "raw" / "android_control"))
        self.config.setdefault("repo_id", "aliaagheis/android-control")
        # REFINED [old]: extracted alongside benchmark_path → [new]: separate extracted_path so read-only inputs (e.g. /kaggle/input) work
        self.config.setdefault("extracted_path", self.config["benchmark_path"])

        self.benchmark_path = Path(self.config['benchmark_path'])
        self.extracted_path = Path(self.config['extracted_path'])

        if not self.benchmark_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                repo_type="dataset",
                local_dir=str(self.config["benchmark_path"]),
                local_dir_use_symlinks=False,
                allow_patterns=["data/android_control.tar.gz"]
            )

        self.zip_path = self.benchmark_path / "data/android_control.tar.gz"

    def load_samples(self) -> None:
        extract_tar(self.zip_path, self.extracted_path)

        parquet_files = sorted(self.extracted_path.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.benchmark_path}")

        # NOTE: cols [episode_id, goal, screenshots_b64, actions, step_instructions]
        # NOTE: screenshots_b64, actions, step_instructions are arrays with len_episode, except screenshots_b64 is len_episode+1 where last image is the output — ignore it
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        df["screenshots_b64"] = df["screenshots_b64"].str[:-1]
        df_exploded = df.explode(["screenshots_b64", "step_instructions", "actions"])
        df_exploded = df_exploded.rename(columns={
            "screenshots_b64": "screenshot_b64",
            "step_instructions": "step_instruction",
            "actions": "action"
        })
        
        self.samples = df_exploded[["episode_id", "goal", "screenshot_b64", "step_instruction", "action"]].to_dict("records")
        self.samples.sort(key=lambda x: x['episode_id'])
        
        self.size = len(self.samples)

    def get_sample(self, idx: int) -> BenchmarkSample:
        "Return a one sample"
        assert idx < self.size
        sample: dict = self.samples[idx]
        # load image
        img = Image.open(BytesIO(base64.b64decode(sample["screenshot_b64"]))).convert('RGB')
        width, height = img.size
        
        # FIX: need h, w
        annotation = {k: v for k, v in sample.items() if k != "screenshot_b64"}
        annotation.update({'w': width, 'h': height})
        return BenchmarkSample(
            screenshot=img,
            task=sample['step_instruction'],
            annotation=annotation,
        )
    
    def score(self, predictions: [AgentOutput]) -> [dict]:
        """Return per-sample metrics. """
        raise NotImplementedError()
    
    @staticmethod
    def score_annotation(self, prediction, annotation) -> [dict]:
        # Maps CustomActionTypes values → AndroidControl GT action_type strings
        _CUSTOM_TO_GT: dict[str, str] = {
            "press_home":  "navigate_home",
            "press_back":  "navigate_back",
            "long_press":  "long_press",
            "scroll":      "scroll",
            "swipe":       "swipe",
            "open_app":    "open_app",
            "terminate":   "status",
            "wait":        "wait",
        }

    @staticmethod
    def visualize_episode(episode: dict):
        imgs  = [Image.open(BytesIO(base64.b64decode(b))) for b in episode["screenshots_b64"]]
        steps =   list(episode["step_instructions"]) + ["end"]
        n     = len(imgs)

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 9))
        if n == 1:
            axes = [axes]

        fig.patch.set_facecolor("black")
        fig.suptitle(
            textwrap.fill(f"Android Control  ·  {episode['episode_id']}  ·  {episode['goal']}", 110),
            color="white", fontsize=9, y=1.01,
        )

        for i, (ax, img, step) in enumerate(zip(axes, imgs,  steps)):
            ax.imshow(img)
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xlabel(
                textwrap.fill(f"{step}", 30),
                color="white", fontsize=7, labelpad=6,
            )

        plt.tight_layout()
        plt.show()