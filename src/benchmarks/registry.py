from benchmarks.base import Benchmark
from benchmarks.screenspotpro import ScreenSpotProBenchmark


BENCHMARKS_REGISTRY: dict[str, type[Benchmark]] = {
    "screenspotpro": ScreenSpotProBenchmark,
}


def build_benchmark(benchmark_config: dict) -> Benchmark:
    cls = BENCHMARKS_REGISTRY[benchmark_config["class"]]
    benchmark = cls(benchmark_config.get("params", {}))
    benchmark.load_samples()
    return benchmark