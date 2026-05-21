import time
from pathlib import Path


def make_run_dir(output_root: str, strategy: str, project_name: str = "llamagen_ucb", experiment_name: str = None) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{strategy}" if not experiment_name else f"{ts}_{strategy}_{experiment_name}"
    run_dir = Path(output_root) / project_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "evals", "generations", "plots"):
        (run_dir / sub).mkdir(exist_ok=True)
    return run_dir
