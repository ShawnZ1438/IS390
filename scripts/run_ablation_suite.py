import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List


_DEFAULT_CONFIGS = [
    "configs/finetune_only.yaml",
    "configs/replay_baseline.yaml",
    "configs/replay_plus_prototypes.yaml",
    "configs/full_inemo_proxy.yaml",
]


def _parse_summary_path(stdout: str) -> str:
    matches = re.findall(r"Saved summary:\s*(.+run_summary\.json)", stdout)
    if not matches:
        raise RuntimeError("Could not find run_summary.json path in run output.")
    return matches[-1].strip()


def _run_one(config_path: str, python_exe: str) -> Dict[str, str]:
    cmd = [python_exe, "-m", "src.run", "--config", config_path]
    started = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Run failed for {config_path} with exit code {proc.returncode}.")

    summary_path = _parse_summary_path(proc.stdout)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    return {
        "label": os.path.splitext(os.path.basename(config_path))[0],
        "config_path": os.path.abspath(config_path),
        "summary_path": os.path.abspath(summary_path),
        "run_dir": os.path.abspath(summary["out_dir"]),
        "run_id": summary["run_id"],
        "duration_sec": round(time.time() - started, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=_DEFAULT_CONFIGS)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--aggregate_out_dir",
        default=None,
        help="Optional directory where aggregate figures and tables will be written.",
    )
    args = ap.parse_args()

    manifest_dir = os.path.dirname(os.path.abspath(args.manifest_out))
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    runs: List[Dict[str, str]] = []
    for config_path in args.configs:
        print(f"=== Running {config_path} ===")
        runs.append(_run_one(config_path, args.python))

    manifest = {
        "created_at_unix": time.time(),
        "python": args.python,
        "runs": runs,
    }
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {args.manifest_out}")

    if args.aggregate_out_dir:
        cmd = [
            args.python,
            "scripts/aggregate_suite.py",
            "--manifest",
            args.manifest_out,
            "--out_dir",
            args.aggregate_out_dir,
        ]
        subprocess.run(cmd, cwd=os.getcwd(), check=True)


if __name__ == "__main__":
    main()
