#!/usr/bin/env python3
"""Run the cold-start runs for L=10 and L=16.

This wrapper calls the copied `run_minGRU_skipconnection.py` file in this folder
with the published hyperparameters.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

RUNS = [
    {
        "L": 10,
        "num_layers": 6,
        "dh": 512,
        "dmodel": 512,
        "lr": 5e-4,
        "lrdecaytime": 5000,
        "RNNsymmetry": "c4v",
        "numsamples": 200,
        "num_epochs": 150000,
        "patch_x": 2,
        "patch_y": 2,
    },
    {
        "L": 16,
        "num_layers": 6,
        "dh": 512,
        "dmodel": 512,
        "lr": 5e-4,
        "lrdecaytime": 5000,
        "RNNsymmetry": "c4v",
        "numsamples": 200,
        "num_epochs": 150000,
        "patch_x": 2,
        "patch_y": 2,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cold-start runs.")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-L", default="", help="Comma-separated filter, e.g. 10 or 10,16")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional override for testing shorter runs.",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runner = here / "run_minGRU_skipconnection.py"

    only = set()
    if args.only_L:
        only = {int(x.strip()) for x in args.only_L.split(",") if x.strip()}

    selected = [cfg.copy() for cfg in RUNS if not only or cfg["L"] in only]
    if args.num_epochs is not None:
        for cfg in selected:
            cfg["num_epochs"] = args.num_epochs

    if not selected:
        raise ValueError("No runs selected.")

    print("[coldstart] selected L:", [cfg["L"] for cfg in selected])

    for cfg in selected:
        cmd = [args.python_bin, str(runner)]
        for k, v in cfg.items():
            cmd.extend([f"--{k}", str(v)])

        print("\n[coldstart]", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=here, check=True)

    print("\n[coldstart] done")


if __name__ == "__main__":
    main()
