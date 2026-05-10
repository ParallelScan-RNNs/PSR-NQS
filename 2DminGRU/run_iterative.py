#!/usr/bin/env python3
"""Single-file iterative retraining runner with per-L campaign overrides.

This file replaces the previous two-file iterative setup: it contains both
the training implementation and the campaign wrapper. It still imports the
shared model and helper modules (`model.py`, `Helper_functions.py`).
"""

from __future__ import annotations

import os
import math
import time
import pickle
import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit
import pandas as pd

from Helper_functions import *          # local_energy, str2bool, ...
from model import *      # TwoDFastGRU, ...

sim_dtype = jnp.float32


# ============================================================
# Schedules
# ============================================================

def nsteps_schedule(L: int, s: float, r: float, L0: int, C: float, F: float,
                        min_steps: int, round_mode: str) -> int:
    """ N_steps(L) = s * [ C exp(-r (L-L0)) + F ]"""
    val = s * (C * math.exp(-r * (L - L0)) + F)
    if round_mode == "ceil":
        n = math.ceil(val)
    elif round_mode == "round":
        n = int(round(val))
    elif round_mode == "floor":
        n = math.floor(val)
    else:
        raise ValueError(f"Unknown round_mode={round_mode}")
    return max(int(min_steps), int(n))


def lr_schedule_decay(step: int, lr0: float, delta: float) -> float:
    # gamma(t) = gamma0 * (1 + t/delta)^(-1)
    return lr0 / (1.0 + step / delta)


def make_optimizer(schedule_fn, *, grad_clip_by_value: bool = False, grad_clip_value: float = 1.0):
    transforms = []
    if grad_clip_by_value:
        transforms.append(optax.clip(grad_clip_value))
    transforms.append(optax.adam(learning_rate=schedule_fn))
    return optax.chain(*transforms)


# ============================================================
# Patch compatibility helper
# ============================================================

def effective_patch(L: int, px: int, py: int):
    """Choose patch sizes that always divide L (avoids reshape dims 0)."""
    px_eff = math.gcd(L, px)
    py_eff = math.gcd(L, py)
    return max(1, px_eff), max(1, py_eff)


# ============================================================
# Checkpoint helpers
# ============================================================

def make_savename(args, px_eff: int, py_eff: int, *, symmetry_tag: str | None = None) -> str:
    if symmetry_tag is None:
        symmetry_tag = args.RNNsymmetry
    return (
        "_px" + str(px_eff)
        + "_py" + str(py_eff)
        + "_dh" + str(args.dh)
        + "_dm" + str(args.dmodel)
        + "_numsamples" + str(args.numsamples)
        + "_RNNsym" + str(symmetry_tag)
    )


def format_scale_tag(scale: float) -> str:
    # Stable compact float formatting for filenames (e.g., 1.0 -> "1", 0.0005 -> "0.0005").
    return format(float(scale), ".12g")


def ckpt_file(saving_path: str, savename: str, *, scale_s: float | None = None) -> str:
    scale_part = f"_scale{format_scale_tag(scale_s)}" if scale_s is not None else ""
    return os.path.join(saving_path, "Params", f"checkpoint{savename}{scale_part}.pkl")


def final_symmetry_for_L(L: int, args) -> str:
    # Current iterative schedule always ends each L with c4v.
    return "c4v"


def ckpt_candidates_for_L(L: int, args, saving_path: str, px_eff: int, py_eff: int):
    # New naming uses final symmetry; legacy naming uses args.RNNsymmetry.
    names = [
        make_savename(args, px_eff, py_eff, symmetry_tag=final_symmetry_for_L(L, args)),
        make_savename(args, px_eff, py_eff, symmetry_tag=args.RNNsymmetry),
    ]
    # Keep order, remove duplicates.
    uniq = []
    seen = set()
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        uniq.append(n)

    # Prefer new scaled checkpoint naming; keep legacy (no-scale) fallback for resume.
    candidates = []
    seen_paths = set()
    for savename in uniq:
        for path in (
            ckpt_file(saving_path, savename, scale_s=args.scale_s),
            ckpt_file(saving_path, savename),
        ):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            candidates.append((savename, path))
    return candidates


def select_existing_ckpt(candidates):
    for savename, path in candidates:
        if os.path.exists(path):
            return savename, path
    return None, None


def save_ckpt(path: str, params, rng_key, epoch, energies, variances, durations, meta: dict, optimizer_state=None):
    ckpt = {
        "model_state": params,
        "optimizer_state": optimizer_state,
        "rng": rng_key,
        "epoch": epoch,
        "energies": energies,
        "variances": variances,
        "durations": durations,
        "meta": meta,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def load_ckpt(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# VMC loss + step
# ============================================================

def get_loss(params, key, numsamples, Nx, Ny, model, RNNsymmetry):
    samples = model.apply(params, key, numsamples, Nx, Ny, method="sample")

    # enforce (numsamples, Nx, Ny)
    if samples.ndim == 2:
        samples = samples[None, :, :]

    if RNNsymmetry == "nosym":
        log_probs = model.apply(params, samples)
    elif RNNsymmetry == "c4v":
        log_probs = model.apply(params, samples, method="logprobs_c4vsym")
    else:
        raise ValueError(f"Unknown RNNsymmetry={RNNsymmetry}")

    e_loc = local_energy(samples, params, model, 0.5 * log_probs, RNNsymmetry)
    e_loc = jnp.real(e_loc)  # energy should be real
    e_loc = jax.lax.stop_gradient(e_loc)

    e_avg = e_loc.mean()
    loss = jnp.mean(log_probs * e_loc - e_avg * log_probs)
    return loss, e_loc


def build_step(model, optimizer, numsamples, Nx, Ny, RNNsymmetry):
    @jit
    def step(params, rng_key, opt_state):
        rng_key, new_key = jax.random.split(rng_key)
        (loss, e_loc), grads = jax.value_and_grad(get_loss, has_aux=True)(
            params, new_key, numsamples, Nx, Ny, model, RNNsymmetry
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, e_loc, new_key
    return step


def final_energy(params, key, model, Nx, Ny, numsamples, RNNsymmetry):
    samples = model.apply(params, key, numsamples, Nx, Ny, method="sample")
    if samples.ndim == 2:
        samples = samples[None, :, :]

    if RNNsymmetry == "nosym":
        log_probs = model.apply(params, samples)
    elif RNNsymmetry == "c4v":
        log_probs = model.apply(params, samples, method="logprobs_c4vsym")
    else:
        raise ValueError(f"Unknown RNNsymmetry={RNNsymmetry}")

    eloc = local_energy(samples, params, model, 0.5 * log_probs, RNNsymmetry)
    eloc = jnp.real(eloc)

    meanE = jnp.mean(eloc)
    varE = jnp.var(eloc)
    err = jnp.sqrt(varE / eloc.shape[0])
    return meanE, varE, err


def final_energy_minibatch_adaptive(
    params,
    key,
    model,
    Nx,
    Ny,
    numsamples_total,
    RNNsymmetry,
    *,
    minibatch_init: int = 2048,
    minibatch_min: int = 64,
    max_retries_per_batch: int = 10,
    verbose: bool = True,
):
    """
    Compute final energy using many total samples but in mini-batches.

    Returns:
      meanE, varE, err, used_total, used_minibatch_final
    """

    # Welford streaming stats for variance
    n_total = 0
    mean = 0.0
    M2 = 0.0

    mb = int(minibatch_init)
    mb_min = int(minibatch_min)
    remaining = int(numsamples_total)

    rng = key

    def update_welford(eloc_batch, n_total, mean, M2):
        eloc_batch = jnp.real(eloc_batch)
        m = int(eloc_batch.shape[0])
        batch_mean = float(jnp.mean(eloc_batch))
        batch_var = float(jnp.var(eloc_batch))  # population var

        if n_total == 0:
            return m, batch_mean, batch_var * m

        n_total_new = n_total + m
        delta = batch_mean - mean
        mean_new = mean + delta * (m / n_total_new)

        batch_M2 = batch_var * m
        M2_new = M2 + batch_M2 + delta * delta * (n_total * m / n_total_new)
        return n_total_new, mean_new, M2_new

    while remaining > 0:
        m = min(mb, remaining)
        retries = 0

        while True:
            try:
                rng, k_samp = jax.random.split(rng)

                samples = model.apply(params, k_samp, m, Nx, Ny, method="sample")
                if samples.ndim == 2:
                    samples = samples[None, :, :]

                if RNNsymmetry == "nosym":
                    log_probs = model.apply(params, samples)
                elif RNNsymmetry == "c4v":
                    log_probs = model.apply(params, samples, method="logprobs_c4vsym")
                else:
                    raise ValueError(f"Unknown RNNsymmetry={RNNsymmetry}")

                eloc = local_energy(samples, params, model, 0.5 * log_probs, RNNsymmetry)
                eloc = jnp.real(eloc)

                n_total, mean, M2 = update_welford(eloc, n_total, mean, M2)
                remaining -= m
                break

            except Exception as e:
                retries += 1
                if verbose:
                    print(
                        f"[final_energy_minibatch_adaptive] batch m={m} failed (mb={mb}). "
                        f"Retry {retries}/{max_retries_per_batch}. Error: {type(e).__name__}: {e}"
                    )

                mb = mb // 2
                if mb < mb_min or retries >= max_retries_per_batch:
                    raise RuntimeError(
                        f"final_energy_minibatch_adaptive failed: cannot fit minibatch >= {mb_min} "
                        f"(last mb tried={max(mb,1)}, requested total={numsamples_total}, processed={n_total})."
                    ) from e

                m = min(mb, remaining)

    var = M2 / n_total if n_total > 0 else float("nan")
    err = math.sqrt(var / n_total) if n_total > 0 else float("nan")

    return (
        jnp.asarray(mean, dtype=jnp.float32),
        jnp.asarray(var, dtype=jnp.float32),
        jnp.asarray(err, dtype=jnp.float32),
        int(n_total),
        int(mb),
    )


# ============================================================
# Stage runner
# ============================================================

def run_stage(*, L, stage_idx, n_stages, stage_name, n_steps, lr_mode, lr_fixed, lr0, delta, symmetry,
              args, model, params, rng_key, all_energies, all_variances, all_durations,
              start_it=0, opt_state_init=None, save_progress_cb=None):
    Nx = Ny = L

    if lr_mode == "fixed":
        schedule_fn = lambda step: lr_fixed
    elif lr_mode == "decay":
        schedule_fn = lambda step: lr_schedule_decay(step, lr0, delta)
    else:
        raise ValueError(f"Unknown lr_mode={lr_mode}")

    optimizer = make_optimizer(
        schedule_fn,
        grad_clip_by_value=args.grad_clip_by_value,
        grad_clip_value=args.grad_clip_value,
    )
    if opt_state_init is None:
        opt_state = optimizer.init(params)
    else:
        opt_state = opt_state_init

    step_fn = build_step(model, optimizer, args.numsamples, Nx, Ny, symmetry)

    print("\n" + "-" * 70)
    print(
        f"[L={L}] {stage_name} ({stage_idx + 1}/{n_stages})"
        f" | steps={n_steps} | start_it={start_it} | symmetry={symmetry} | lr_mode={lr_mode}"
        f" | grad_clip_by_value={args.grad_clip_by_value}"
        f" | grad_clip_value={args.grad_clip_value}"
    )
    print("-" * 70)

    for it in range(start_it, n_steps):
        t0 = time.perf_counter()
        params, opt_state, loss, eloc, rng_key = step_fn(params, rng_key, opt_state)
        t1 = time.perf_counter()

        all_energies.append(float(jnp.mean(eloc)))
        all_variances.append(float(jnp.var(eloc)))
        all_durations.append(t1 - t0)

        if (it % args.print_every) == 0 or (it == n_steps - 1):
            print(
                f"[L={L} {stage_name}] it={it:7d}/{n_steps}"
                f"  E={all_energies[-1]}  Var={all_variances[-1]}  dt={all_durations[-1]}"
            )

        if save_progress_cb is not None and args.ckpt_every > 0:
            should_save = ((it + 1) % args.ckpt_every == 0) or (it == n_steps - 1)
            if should_save:
                save_progress_cb(
                    params=params,
                    rng_key=rng_key,
                    optimizer_state=opt_state,
                    stage_idx=stage_idx,
                    stage_name=stage_name,
                    stage_step=it + 1,
                    stage_n_steps=n_steps,
                )

    return params, rng_key, opt_state


# ============================================================
# Main training per L
# ============================================================

def train_L(L: int, args, *, warmstart_ckpt_path: str | None):
    Nx = Ny = L

    px_eff, py_eff = effective_patch(L, args.patch_x, args.patch_y)

    saving_path = os.path.join(args.base_dir, f"L_{L}", f"dh_{args.dh}", f"numlayers_{args.num_layers}")
    os.makedirs(os.path.join(saving_path, "Outputs"), exist_ok=True)
    os.makedirs(os.path.join(saving_path, "Params"), exist_ok=True)

    ckpt_candidates = ckpt_candidates_for_L(L, args, saving_path, px_eff, py_eff)
    savename, ckpt_curr = ckpt_candidates[0]

    if args.enforce_equal_dims and (args.dmodel != args.dh):
        raise ValueError(
            f"dmodel ({args.dmodel}) != dh ({args.dh}). "
            f"Your current model adds x + skip_input, so require dmodel==dh "
            f"OR edit model.py to project skip to d_model."
        )

    model = TwoDFastGRU(
        d_hidden=args.dh,
        d_model=args.dmodel,
        n_layers=args.num_layers,
        patch_x=px_eff,
        patch_y=py_eff,
    )

    rng_key = jax.random.key(args.seed)

    params = None
    all_energies, all_variances, all_durations = [], [], []
    resume_stage_idx = 0
    resume_stage_step = 0
    resume_opt_state = None

    if args.resume:
        existing_savename, existing_ckpt = select_existing_ckpt(ckpt_candidates)
    else:
        existing_savename, existing_ckpt = None, None

    if existing_ckpt is not None:
        savename = existing_savename
        ckpt_curr = existing_ckpt
        print(f"[Resume] Loading current L checkpoint: {ckpt_curr}")
        ckpt = load_ckpt(ckpt_curr)
        params = ckpt["model_state"]
        rng_key = ckpt["rng"]
        all_energies = list(ckpt.get("energies", []))
        all_variances = list(ckpt.get("variances", []))
        all_durations = list(ckpt.get("durations", []))
        resume_opt_state = ckpt.get("optimizer_state")
        meta = ckpt.get("meta", {})
        progress = meta.get("progress", {})
        resume_stage_idx = int(progress.get("stage_idx", 0))
        resume_stage_step = int(progress.get("stage_step", 0))
        status = meta.get("status")
        if status is None:
            # Backward compatibility: older checkpoints in this script were final-only saves.
            status = "complete" if ("final_eval" in meta) else "in_progress"

        if status == "complete":
            print(f"[L={L}] checkpoint already complete. Skipping.")
            return {"completed": True, "ckpt_path": ckpt_curr}
    elif warmstart_ckpt_path is not None and os.path.exists(warmstart_ckpt_path):
        print(f"[Warm-start] Loading previous checkpoint: {warmstart_ckpt_path}")
        ckpt = load_ckpt(warmstart_ckpt_path)
        params = ckpt["model_state"]
        rng_key = ckpt["rng"]

    if params is None:
        rng_key, init_key = jax.random.split(rng_key)
        dummy = jnp.zeros((1, L, L), dtype=jnp.int32)
        params = model.init(init_key, dummy)

    if not args.dotraining:
        print("dotraining=False -> skip training.")
        return {"completed": False, "ckpt_path": ckpt_curr}

    if L == args.L0:
        s = args.scale_s
        delta = args.delta0 * s
        stages = [
            {
                "stage_name": "stage1",
                "n_steps": int(args.stage1_base + args.stage1_scale * s),
                "lr_mode": "fixed",
                "lr_fixed": args.lr_stage1,
                "lr0": args.lr0_stage,
                "delta": delta,
                "symmetry": "nosym",
            },
            {
                "stage_name": "stage2",
                "n_steps": int(args.stage2_steps),
                "lr_mode": "decay",
                "lr_fixed": args.lr_stage1,
                "lr0": args.lr0_stage,
                "delta": delta,
                "symmetry": "nosym",
            },
            {
                "stage_name": "stage3",
                "n_steps": int(args.stage3_steps),
                "lr_mode": "decay",
                "lr_fixed": args.lr_stage1,
                "lr0": args.lr0_stage,
                "delta": delta,
                "symmetry": "c4v",
            },
        ]
    else:
        stages = [
            {
                "stage_name": "L>6",
                "n_steps": nsteps_schedule(
                    L=L, s=args.scale_s, r=args.rate_r, L0=args.L0, C=args.C, F=args.F,
                    min_steps=args.min_steps, round_mode=args.round_mode
                ),
                "lr_mode": "fixed",
                "lr_fixed": args.lr_large,
                "lr0": args.lr0_stage,
                "delta": args.delta0 * args.scale_s,
                "symmetry": "c4v",
            }
        ]
    final_sym = "c4v"

    def save_progress_checkpoint(*, params, rng_key, optimizer_state, stage_idx, stage_name, stage_step, stage_n_steps):
        meta = {
            "status": "in_progress",
            "L": L,
            "patch_effective": {"px": px_eff, "py": py_eff},
            "progress": {
                "stage_idx": int(stage_idx),
                "stage_name": str(stage_name),
                "stage_step": int(stage_step),
                "stage_n_steps": int(stage_n_steps),
                "num_stages": int(len(stages)),
                "global_step": int(len(all_energies)),
            },
        }
        save_ckpt(
            ckpt_curr,
            params=params,
            rng_key=rng_key,
            epoch=len(all_energies),
            energies=all_energies,
            variances=all_variances,
            durations=all_durations,
            meta=meta,
            optimizer_state=optimizer_state,
        )

    if resume_stage_idx >= len(stages):
        print(f"[L={L}] training stages already finished, continuing to final evaluation/writeout.")
    else:
        for stage_idx, stage in enumerate(stages):
            if stage_idx < resume_stage_idx:
                continue
            start_it = resume_stage_step if stage_idx == resume_stage_idx else 0
            opt_state_init = resume_opt_state if (stage_idx == resume_stage_idx and start_it > 0) else None

            params, rng_key, _ = run_stage(
                L=L,
                stage_idx=stage_idx,
                n_stages=len(stages),
                stage_name=stage["stage_name"],
                n_steps=stage["n_steps"],
                lr_mode=stage["lr_mode"],
                lr_fixed=stage["lr_fixed"],
                lr0=stage["lr0"],
                delta=stage["delta"],
                symmetry=stage["symmetry"],
                args=args,
                model=model,
                params=params,
                rng_key=rng_key,
                all_energies=all_energies,
                all_variances=all_variances,
                all_durations=all_durations,
                start_it=start_it,
                opt_state_init=opt_state_init,
                save_progress_cb=save_progress_checkpoint,
            )
            resume_stage_step = 0
            resume_opt_state = None

        # Save a post-training (pre-final-eval) checkpoint to survive evaluation failures.
        save_progress_checkpoint(
            params=params,
            rng_key=rng_key,
            optimizer_state=None,
            stage_idx=len(stages),
            stage_name="post_training",
            stage_step=0,
            stage_n_steps=0,
        )

    # Final evaluation in adaptive mini-batches
    eval_key, rng_key = jax.random.split(rng_key)
    meanE_fin, varE_fin, err_fin, used_total, used_mb = final_energy_minibatch_adaptive(
        params, eval_key, model, Nx, Ny,
        args.final_numsamples,
        final_sym,
        minibatch_init=args.final_minibatch,
        minibatch_min=args.final_minibatch_min,
        max_retries_per_batch=args.final_max_retries_per_batch,
        verbose=True,
    )
    print(f"[FINAL L={L}] E={meanE_fin} Var={varE_fin} Err={err_fin} (total={used_total}, mb={used_mb})")

    out_csv = os.path.join(saving_path, "Outputs", f"outputs{savename}.csv")
    df = pd.DataFrame({"Energies": all_energies, "Variance": all_variances, "Time": all_durations})
    df.to_csv(out_csv, index=False)

    np.savetxt(os.path.join(saving_path, f"energies{savename}.txt"), np.array(all_energies))
    np.savetxt(os.path.join(saving_path, f"variances{savename}.txt"), np.array(all_variances))

    # FIXED: write used_total/used_mb (not used_n)
    with open(os.path.join(saving_path, "Outputs", f"final_energy{savename}.txt"), "w") as f:
        f.write(
            f"symmetry {final_sym}\n"
            f"numsamples_total_requested {int(args.final_numsamples)}\n"
            f"numsamples_total_used {int(used_total)}\n"
            f"minibatch_init {int(args.final_minibatch)}\n"
            f"minibatch_min {int(args.final_minibatch_min)}\n"
            f"minibatch_used {int(used_mb)}\n"
            f"meanE {float(meanE_fin)}\n"
            f"varE {float(varE_fin)}\n"
            f"err {float(err_fin)}\n"
        )

    meta = {
        "L": L,
        "patch_effective": {"px": px_eff, "py": py_eff},
        "eq6": {"s": args.scale_s, "r": args.rate_r, "L0": args.L0, "C": args.C, "F": args.F},
        "stage_params": {
            "lr_stage1": args.lr_stage1,
            "lr0_stage": args.lr0_stage,
            "delta0": args.delta0,
            "lr_large": args.lr_large,
            "grad_clip_by_value": bool(args.grad_clip_by_value),
            "grad_clip_value": float(args.grad_clip_value),
            "stage1_base": args.stage1_base,
            "stage1_scale": args.stage1_scale,
            "stage2_steps": args.stage2_steps,
            "stage3_steps": args.stage3_steps,
        },
        "final_eval": {
            "symmetry": final_sym,
            "numsamples_total_requested": int(args.final_numsamples),
            "numsamples_total_used": int(used_total),
            "minibatch_init": int(args.final_minibatch),
            "minibatch_min": int(args.final_minibatch_min),
            "minibatch_used": int(used_mb),
            "meanE": float(meanE_fin),
            "varE": float(varE_fin),
            "err": float(err_fin),
        }
    }

    save_ckpt(
        ckpt_curr, params, rng_key, epoch=len(all_energies),
        energies=all_energies, variances=all_variances, durations=all_durations,
        meta={**meta, "status": "complete", "progress": {"stage_idx": len(stages), "stage_name": "complete", "stage_step": 0}},
        optimizer_state=None,
    )
    print(f"[Saved] {ckpt_curr}")
    return {"completed": True, "ckpt_path": ckpt_curr}


def find_latest_checkpoint(args):
    for L in range(args.L_end, args.L_start - 1, -args.L_stride):
        px_eff, py_eff = effective_patch(L, args.patch_x, args.patch_y)
        saving_path = os.path.join(args.base_dir, f"L_{L}", f"dh_{args.dh}", f"numlayers_{args.num_layers}")
        _, path = select_existing_ckpt(ckpt_candidates_for_L(L, args, saving_path, px_eff, py_eff))
        if path is None:
            continue
        try:
            ckpt = load_ckpt(path)
            meta = ckpt.get("meta", {})
            status = meta.get("status")
            if status not in {"complete", "in_progress"}:
                # Backward compatibility: old checkpoints in this script were saved only at the end.
                status = "complete"
            return {"L": L, "path": path, "status": status}
        except Exception as e:
            print(f"[Resume] Warning: failed to inspect checkpoint at L={L}: {type(e).__name__}: {e}")
            continue
    return None

# ============================================================
# Campaign driver: single-file iterative retraining entry point
# ============================================================

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
    """Build the full argument namespace expected by train_L(...)."""
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


def select_campaign(l_min: int | None, l_max: int | None) -> list[LConfig]:
    selected = []
    for cfg in CAMPAIGN:
        if l_min is not None and cfg.L < l_min:
            continue
        if l_max is not None and cfg.L > l_max:
            continue
        selected.append(cfg)
    if not selected:
        raise ValueError("No L values selected.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run iterative-retraining runs from a single file.")
    parser.add_argument("--base-dir", default="./TwoDminGRU_iter_repro")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--l-min", type=int, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    selected = select_campaign(args.l_min, args.l_max)
    print("[iterative] selected L:", [c.L for c in selected])

    warmstart = None
    for cfg in selected:
        run_args = build_args(args.base_dir, cfg, args.seed, resume=not args.no_resume)
        print(f"\n[iterative] L={cfg.L} s={cfg.scale_s} F={cfg.F} lr_iter={cfg.lr_iter}")
        print("[iterative] warmstart:", warmstart if warmstart else "<none>")

        if args.dry_run:
            px_eff, py_eff = effective_patch(cfg.L, run_args.patch_x, run_args.patch_y)
            save_root = Path(run_args.base_dir) / f"L_{cfg.L}" / f"dh_{run_args.dh}" / f"numlayers_{run_args.num_layers}"
            savename = make_savename(run_args, px_eff, py_eff, symmetry_tag=final_symmetry_for_L(cfg.L, run_args))
            warmstart = str(ckpt_file(str(save_root), savename, scale_s=run_args.scale_s))
            print("[iterative] planned checkpoint:", warmstart)
            continue

        result = train_L(cfg.L, run_args, warmstart_ckpt_path=warmstart)
        warmstart = result.get("ckpt_path") if isinstance(result, dict) else None

    print("\n[iterative] done")


if __name__ == "__main__":
    main()
