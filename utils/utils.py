from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataArgs:
    train: str
    max_samples: int | None = None
    poisoned: bool = False


def setup_directories(run_name: str, base_dir: Path) -> dict:
    """Create directory structure for the run."""
    run_dir = base_dir / run_name
    dirs = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "final": run_dir / "final",
        "logs": run_dir / "logs",
        "configs": run_dir / "configs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
