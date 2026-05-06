#!/usr/bin/env python3
"""Run iterative retraining with per-L hyperparameters.

This wrapper imports the copied `run_minGRU_iterative_retraining.py` file in
this folder and calls `train_L(...)` directly so per-L overrides are explicit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass(frozen=True)
class LConfig:
    L: int
    scale_s: float
    F: float
    lr_iter: float


CAMPAIGN = [
    LConfig(6, 4.0, 2000.0, 5e-5),
    LConfig(8, 4.0, 2000.0, 5e-5),
    LConfig(10, 4.0, 2000.0, 5e-5),
    LConfig(12, 4.0, 2000.0, 5e-5),
    LConfig(14, 4.0, 2000.0, 5e-5),
    LConfig(16, 4.0, 2000.0, 5e-5),
    LConfig(18, 1.0, 2000.0, 1e-4),
    LConfig(20, 1.0, 2000.0, 1e-4),
    LConfig(22, 1.0, 2000.0, 1e-4),
    LConfig(24, 1.0, 2000.0, 1e-4),
    LConfig(26, 1.0, 2000.0, 1e-4),
    LConfig(28, 1.0, 2000.0, 1e-4),
    LConfig(30, 1.0, 2000.0, 1e-4),
    LConfig(32, 1.0, 2000.0, 1e-4),
    LConfig(36, 1.0, 1000.0, 1e-4),
    LConfig(40, 1.0, 1000.0, 1e-4),
    LConfig(44, 1.0, 1000.0, 1e-4),
    LConfig(46, 1.0, 500.0, 1e-4),
    LConfig(48, 1.0, 500.0, 1e-4),
    LConfig(50, 1.0, 500.0, 1e-4),
]


def build_args(base_dir: str, cfg: LConfig, seed: int, resume: bool) -> SimpleNamespace:
    return SimpleNamespace(
        num_layers=3,
        dh=256,
        dmodel=256,
        RNNsymmetry="c4v",
        numsamples=200,
        patch_x=2,
        patch_y=2,
        dotraining=True,
        enforce_equal_dims=True,
        L_start=cfg.L,
        L_end=cfg.L,
        L_stride=2,
        scale_s=cfg.scale_s,
        rate_r=0.25,
        L0=6,
        C=101000.0,
        F=cfg.F,
        min_steps=200,
        round_mode="ceil",
        lr_stage1=5e-4,
        lr0_stage=5e-4,
        delta0=5000.0,
        lr_large=cfg.lr_iter,
        grad_clip_by_value=False,
        grad_clip_value=1.0,
        stage1_base=1000.0,
        stage1_scale=50000.0,
        stage2_steps=76000.0,
        stage3_steps=101000.0,
        print_every=200,
        base_dir=base_dir,
        resume=resume,
        resume_from_latest=False,
        ckpt_every=1000,
        seed=seed,
        print_schedule=False,
        final_numsamples=100000,
        final_minibatch=2048,
        final_minibatch_min=64,
        final_max_retries_per_batch=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative-retraining runs.")
    parser.add_argument("--base-dir", default="./TwoDminGRU_iter_repro")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--l-min", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import run_minGRU_iterative_retraining as core  # noqa: WPS433

    selected = []
    for cfg in CAMPAIGN:
        if args.l_min is not None and cfg.L < args.l_min:
            continue
        if args.l_max is not None and cfg.L > args.l_max:
            continue
        selected.append(cfg)

    if not selected:
        raise ValueError("No L values selected.")

    print("[iterative] selected L:", [c.L for c in selected])
    warmstart = None

    for cfg in selected:
        run_args = build_args(args.base_dir, cfg, args.seed, resume=not args.no_resume)
        print(f"\n[iterative] L={cfg.L} s={cfg.scale_s} F={cfg.F} lr_iter={cfg.lr_iter}")
        print("[iterative] warmstart:", warmstart if warmstart else "<none>")

        if args.dry_run:
            px_eff, py_eff = core.effective_patch(cfg.L, run_args.patch_x, run_args.patch_y)
            save_root = Path(run_args.base_dir) / f"L_{cfg.L}" / f"dh_{run_args.dh}" / f"numlayers_{run_args.num_layers}"
            savename = core.make_savename(run_args, px_eff, py_eff, symmetry_tag=core.final_symmetry_for_L(cfg.L, run_args))
            warmstart = str(core.ckpt_file(str(save_root), savename, scale_s=run_args.scale_s))
            continue

        result = core.train_L(cfg.L, run_args, warmstart_ckpt_path=warmstart)
        warmstart = result.get("ckpt_path") if isinstance(result, dict) else None

    print("\n[iterative] done")


if __name__ == "__main__":
    main()
