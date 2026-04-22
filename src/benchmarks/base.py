from src.agents.base import AgentOutput

from PIL.Image import Image
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict

class BenchmarkSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    screenshot: Image
    task: str
    annotation: dict


class Benchmark(ABC):
    def __init__(self, config: dict[str, any]) -> None:
        self.config = config
        self.size =  0

    def load_samples(self) -> None:
        pass
    
    @abstractmethod
    def get_sample(self, idx: int) -> BenchmarkSample:
        "Return a one sample"

    def get_sample_from_annotation(self, annotation: dict) -> BenchmarkSample:
        "Return Benchmark Sample from annotation dict (used in rerun failed ones)"
        raise NotImplementedError("Benchmark has no implementation of get sample from annotation")

    @abstractmethod
    def score(self, predictions: [AgentOutput]) -> [dict]:
        """Return per-sample metrics. """
        