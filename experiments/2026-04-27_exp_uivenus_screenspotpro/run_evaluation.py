from pathlib import Path
import subprocess
import sys


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(__file__).resolve().parent / "evaluate_grounder.yaml"

    cmd = [
        sys.executable,
        str(repo_root / "src" / "pipeline" / "evaluate_grounder.py"),
        str(config_path),
    ]
    return subprocess.call(cmd, cwd=str(repo_root))


if __name__ == "__main__":
    raise SystemExit(main())
