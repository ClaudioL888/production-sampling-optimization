"""Run archived experiments in separate working directories."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASKS = {
    "plot": ("optimize1_3.py", ["draw"]),
    "efficacy": ("efficacy_function.py", []),
    "metric": ("v_optimizer.py", []),
    "summarize": ("22.py", []),
    "optimize": ("optimize1_3.py", ["full", "1"]),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    args = parser.parse_args()
    working = ROOT / "runs" / args.task
    working.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "results").glob("*.csv"):
        target = working / source.name
        if not target.exists():
            shutil.copy2(source, target)
    script, arguments = TASKS[args.task]
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print(f"Task: {args.task} | Output: {working}", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "src" / script), *arguments], cwd=working, env=env)

if __name__ == "__main__":
    raise SystemExit(main())
