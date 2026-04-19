from src.benchmarks.base import Benchmark, BenchmarkSample
from src.agents.base import AgentOutput
from src.constants import DATA_PATH

import json
from PIL import Image
from pathlib import Path
from huggingface_hub import snapshot_download


class ScreenSpotProBenchmark(Benchmark):
    def __init__(self, config: dict[str, any]) -> None:
        super().__init__(config)
        self.config.setdefault("benchmark_path", str(DATA_PATH / "raw" / "screenspot_pro"))
        self.config.setdefault("repo_id", "likaixin/ScreenSpot-Pro")
        
        self.benchmark_path = Path(self.config['benchmark_path'])
        if not self.benchmark_path.exists():
            snapshot_download(
                repo_id=self.config["repo_id"],
                repo_type="dataset",
                local_dir=str(self.config["benchmark_path"]),
                local_dir_use_symlinks=False,
            )
        
        self.images_dir_path = self.benchmark_path / "images"
        self.annotations_dir_path = self.benchmark_path / "annotations"
        # list of paths of json files 
        self.annotations_paths = list(self.annotations_dir_path.iterdir())
        
        self.samples: list|None = None
        
    def load_samples(self) -> None:
        annotation_groups = [json.loads(ann_pth.read_text())
                        for ann_pth in self.annotations_paths]
        
        self.samples = [sample for group in annotation_groups for sample in group]
        self.size = len(self.samples)
        
    def get_sample(self, idx: int) -> BenchmarkSample:
        "Return a one sample"
        assert idx < self.size
        
        sample = self.samples[idx]
        # load image
        sample_img_path = self.images_dir_path / sample['img_filename']
        img = Image.open(sample_img_path).convert('RGB')
        
        return BenchmarkSample(
            screenshot=img,
            task=sample['instruction'],
            annotation=sample
        )
    
    def score(self, predictions: [AgentOutput]) -> [dict]:
        """Return per-sample metrics. """
        raise NotImplementedError()