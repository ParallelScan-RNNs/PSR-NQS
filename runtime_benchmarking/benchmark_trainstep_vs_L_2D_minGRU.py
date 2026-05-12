import argparse
import os
import time
import math
import resource
import sys
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from model_skipconnection_opt import TwoDFastGRU # implementation with scan
# from model_skipconnection import TwoDFastGRU #implemntation with for loops instead of scan, to avoid scan overhead in small L regime

sim_dtype = jnp.float32


def parse_l_values(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def effective_patch_dims(nx: int, ny: int, px: int, py: int):
    px_eff = math.gcd(nx, px)
    py_eff = math.gcd(ny, py)
    return max(1, px_eff), max(1, py_eff)


def peak_rss_mb():
    # Linux ru_maxrss is in KB; macOS is bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss / (1024.0 * 1024.0))
    return float(rss / 1024.0)


def gpu_memory_snapshot_mb():
    """
    Return GPU memory snapshot from nvidia-smi.
    Keys:
      gpu_mem_used_mb, gpu_mem_total_mb, gpu_mem_free_mb, gpu_mem_used_frac,
      gpu_proc_mem_mb (for current PID; may be 0 if unavailable)
    Returns None if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    used_total = 0.0
    mem_total = 0.0
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        used_s, total_s = [x.strip() for x in line.split(",")[:2]]
        used_total += float(used_s)
        mem_total += float(total_s)

    proc_mem = 0.0
    try:
        out_proc = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        pid = os.getpid()
        for line in out_proc.strip().splitlines():
            if not line.strip():
                continue
            pid_s, mem_s = [x.strip() for x in line.split(",")[:2]]
            if int(pid_s) == pid:
                proc_mem += float(mem_s)
    except Exception:
        pass

    free_total = max(0.0, mem_total - used_total)
    used_frac = (used_total / mem_total) if mem_total > 0 else float("nan")
    return {
        "gpu_mem_used_mb": float(used_total),
        "gpu_mem_total_mb": float(mem_total),
        "gpu_mem_free_mb": float(free_total),
        "gpu_mem_used_frac": float(used_frac),
        "gpu_proc_mem_mb": float(proc_mem),
    }


def forward_log_probs(model, params, samples, mode: str):
    if mode == "parallel":
        return model.apply(params, samples)
    if mode == "sequential":
        return model.apply(params, samples, method="sequential_call")
    raise ValueError(f"Unknown mode: {mode}")


def local_energy_mode(samples, params, model, log_psi, mode: str):
    numsamples, nx, ny = samples.shape

    local_energies = jnp.zeros((numsamples,), dtype=sim_dtype)

    for i in range(nx - 1):
        spins_products = 0.25 * (2 * samples[:, i] - 1) * (2 * samples[:, i + 1] - 1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis=1)

    for j in range(ny - 1):
        spins_products = 0.25 * (2 * samples[:, :, j] - 1) * (2 * samples[:, :, j + 1] - 1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis=1)

    def step_fn_horizontal(n, state):
        s, output = state
        i = n // ny
        j = n % ny

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i + 1, j].set(1 - flipped_state[:, i + 1, j])
        flipped_logpsi = 0.5 * forward_log_probs(model, params, flipped_state, mode)

        output += (s[:, i, j] + s[:, i + 1, j] == 1) * (-0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        j = n // nx
        i = n % nx

        flipped_state = s.at[:, i, j].set(1 - s[:, i, j])
        flipped_state = flipped_state.at[:, i, j + 1].set(1 - flipped_state[:, i, j + 1])
        flipped_logpsi = 0.5 * forward_log_probs(model, params, flipped_state, mode)

        output += (s[:, i, j] + s[:, i, j + 1] == 1) * (-0.5) * jnp.exp(flipped_logpsi - log_psi)
        return s, output

    output = jnp.zeros((numsamples,), dtype=sim_dtype)
    _, off_diag_v = jax.lax.fori_loop(0, nx * (ny - 1), step_fn_vertical, (samples, output))
    _, off_diag_h = jax.lax.fori_loop(0, (nx - 1) * ny, step_fn_horizontal, (samples, output))

    return local_energies + off_diag_v + off_diag_h


def make_step_fn(model, optimizer, numsamples, nx, ny, mode: str, symmetry: str):
    def get_loss(params, key):
        samples = model.apply(params, key, numsamples, nx, ny, method="sample")
        if symmetry == "nosym":
            log_probs = forward_log_probs(model, params, samples, mode)
        elif symmetry == "c4v":
            log_probs = model.apply(params, samples, method="logprobs_c4vsym")
        else:
            raise ValueError(f"Unknown symmetry: {symmetry}")
        e_loc = jax.lax.stop_gradient(local_energy_mode(samples, params, model, 0.5 * log_probs, mode))
        e_avg = e_loc.mean()
        loss = jnp.mean(log_probs * e_loc - e_avg * log_probs)
        return loss, e_loc

    @jax.jit
    def step(params, opt_state, rng_key):
        rng_key, new_key = jax.random.split(rng_key)
        (loss, e_loc), grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, new_key, loss, e_loc

    return step


def benchmark_for_mode(
    model,
    params,
    optimizer,
    numsamples,
    nx,
    ny,
    repeats,
    warmup_steps,
    mode,
    symmetry,
    print_each_step=False,
    l_size_for_log=None,
    report_gpu_memory=False,
):
    rss_before_mb = peak_rss_mb()
    gpu_before = gpu_memory_snapshot_mb() if report_gpu_memory else None
    opt_state = optimizer.init(params)
    step_fn = make_step_fn(model, optimizer, numsamples, nx, ny, mode, symmetry)
    rng_key = jax.random.key(0)

    # First step includes JIT compile + first execution.
    t0 = time.perf_counter()
    params, opt_state, rng_key, loss, _ = step_fn(params, opt_state, rng_key)
    loss.block_until_ready()
    compile_first_step_s = time.perf_counter() - t0
    if print_each_step:
        print(
            f"[step] L={l_size_for_log} mode={mode} phase=compile_first "
            f"step=1/{max(1, warmup_steps)} dt={compile_first_step_s:.6f}s"
        )
    rss_after_compile_mb = peak_rss_mb()
    gpu_after_compile = gpu_memory_snapshot_mb() if report_gpu_memory else None

    warmup_extra_start = time.perf_counter()
    for i in range(max(0, warmup_steps - 1)):
        wi0 = time.perf_counter()
        params, opt_state, rng_key, loss, _ = step_fn(params, opt_state, rng_key)
        loss.block_until_ready()
        if print_each_step:
            print(
                f"[step] L={l_size_for_log} mode={mode} phase=warmup "
                f"step={i + 2}/{warmup_steps} dt={time.perf_counter() - wi0:.6f}s"
            )
    warmup_extra_s = time.perf_counter() - warmup_extra_start
    rss_after_warmup_mb = peak_rss_mb()

    timed_block_start = time.perf_counter()
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        params, opt_state, rng_key, loss, _ = step_fn(params, opt_state, rng_key)
        loss.block_until_ready()
        dt = time.perf_counter() - start
        times.append(dt)
        if print_each_step:
            print(
                f"[step] L={l_size_for_log} mode={mode} phase=timed "
                f"step={i + 1}/{repeats} dt={dt:.6f}s"
            )
    timed_block_s = time.perf_counter() - timed_block_start
    rss_after_timed_mb = peak_rss_mb()
    gpu_after_timed = gpu_memory_snapshot_mb() if report_gpu_memory else None

    arr = np.asarray(times, dtype=np.float64)
    stats = {
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "std_s": float(np.std(arr)),
        "compile_first_step_s": float(compile_first_step_s),
        "warmup_extra_s": float(warmup_extra_s),
        "timed_block_s": float(timed_block_s),
        "n_timed": int(repeats),
        "peak_rss_before_mb": float(rss_before_mb),
        "peak_rss_after_compile_mb": float(rss_after_compile_mb),
        "peak_rss_after_warmup_mb": float(rss_after_warmup_mb),
        "peak_rss_after_timed_mb": float(rss_after_timed_mb),
        "delta_peak_rss_compile_mb": float(rss_after_compile_mb - rss_before_mb),
        "delta_peak_rss_timed_mb": float(rss_after_timed_mb - rss_after_compile_mb),
    }
    if report_gpu_memory:
        stats.update(
            {
                "gpu_mem_before_mb": None if gpu_before is None else gpu_before["gpu_mem_used_mb"],
                "gpu_mem_after_compile_mb": None if gpu_after_compile is None else gpu_after_compile["gpu_mem_used_mb"],
                "gpu_mem_after_timed_mb": None if gpu_after_timed is None else gpu_after_timed["gpu_mem_used_mb"],
                "gpu_proc_mem_before_mb": None if gpu_before is None else gpu_before["gpu_proc_mem_mb"],
                "gpu_proc_mem_after_compile_mb": None if gpu_after_compile is None else gpu_after_compile["gpu_proc_mem_mb"],
                "gpu_proc_mem_after_timed_mb": None if gpu_after_timed is None else gpu_after_timed["gpu_proc_mem_mb"],
                "gpu_delta_mem_compile_mb": (
                    None
                    if (gpu_before is None or gpu_after_compile is None)
                    else gpu_after_compile["gpu_mem_used_mb"] - gpu_before["gpu_mem_used_mb"]
                ),
                "gpu_delta_mem_timed_mb": (
                    None
                    if (gpu_after_compile is None or gpu_after_timed is None)
                    else gpu_after_timed["gpu_mem_used_mb"] - gpu_after_compile["gpu_mem_used_mb"]
                ),
            }
        )
    return stats


def benchmark_forward_only_for_mode(
    model,
    params,
    samples,
    repeats,
    warmup_steps,
    mode,
    print_each_step=False,
    l_size_for_log=None,
):
    @jax.jit
    def fwd_fn(p, x):
        return forward_log_probs(model, p, x, mode)

    # First call includes compile + first execution.
    t0 = time.perf_counter()
    y = fwd_fn(params, samples)
    y.block_until_ready()
    compile_first_step_s = time.perf_counter() - t0
    if print_each_step:
        print(
            f"[fwd] L={l_size_for_log} mode={mode} phase=compile_first "
            f"step=1/{max(1, warmup_steps)} dt={compile_first_step_s:.6f}s"
        )

    warmup_extra_start = time.perf_counter()
    for i in range(max(0, warmup_steps - 1)):
        wi0 = time.perf_counter()
        y = fwd_fn(params, samples)
        y.block_until_ready()
        if print_each_step:
            print(
                f"[fwd] L={l_size_for_log} mode={mode} phase=warmup "
                f"step={i + 2}/{warmup_steps} dt={time.perf_counter() - wi0:.6f}s"
            )
    warmup_extra_s = time.perf_counter() - warmup_extra_start

    timed_block_start = time.perf_counter()
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        y = fwd_fn(params, samples)
        y.block_until_ready()
        dt = time.perf_counter() - start
        times.append(dt)
        if print_each_step:
            print(
                f"[fwd] L={l_size_for_log} mode={mode} phase=timed "
                f"step={i + 1}/{repeats} dt={dt:.6f}s"
            )
    timed_block_s = time.perf_counter() - timed_block_start

    arr = np.asarray(times, dtype=np.float64)
    return {
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p90_s": float(np.percentile(arr, 90)),
        "std_s": float(np.std(arr)),
        "compile_first_step_s": float(compile_first_step_s),
        "warmup_extra_s": float(warmup_extra_s),
        "timed_block_s": float(timed_block_s),
        "n_timed": int(repeats),
    }


def run_sanity_checks(model, params, l_size, sanity_samples):
    key_data = jax.random.key(123)
    x_rand = jax.random.randint(key_data, (sanity_samples, l_size[0], l_size[1]), 0, 2)

    logp_parallel = forward_log_probs(model, params, x_rand, "parallel")
    logp_sequential = forward_log_probs(model, params, x_rand, "sequential")
    diff_rand = jnp.abs(logp_parallel - logp_sequential)
    print(
        "[Sanity] random-input logp |parallel-sequential|: "
        f"max={float(jnp.max(diff_rand)):.3e}, mean={float(jnp.mean(diff_rand)):.3e}"
    )

    key_sample = jax.random.key(456)
    sampled = model.apply(params, key_sample, sanity_samples, l_size[0], l_size[1], method="sample")
    logp_sample_parallel = forward_log_probs(model, params, sampled, "parallel")
    logp_sample_sequential = forward_log_probs(model, params, sampled, "sequential")
    diff_sample = jnp.abs(logp_sample_parallel - logp_sample_sequential)
    print(
        "[Sanity] sampled-input logp |parallel-sequential|: "
        f"max={float(jnp.max(diff_sample)):.3e}, mean={float(jnp.mean(diff_sample)):.3e}"
    )

    sample_a = model.apply(params, key_sample, sanity_samples, l_size[0], l_size[1], method="sample")
    sample_b = model.apply(params, key_sample, sanity_samples, l_size[0], l_size[1], method="sample")
    deterministic = bool(jnp.array_equal(sample_a, sample_b))
    print(f"[Sanity] sample deterministic for same key: {deterministic}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark one train-step time vs L for parallel vs sequential calls")
    parser.add_argument("--L_values", type=str, default="6,8,10,12")
    parser.add_argument("--geometry", type=str, default="square", choices=["square", "cylinder"])
    parser.add_argument("--cylinder_circumference", type=int, default=2,
                        help="Nx for cylinder geometry; Ny is swept by L_values.")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dh", type=int, default=256)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--numsamples", type=int, default=100)
    parser.add_argument("--patch_x", type=int, default=2)
    parser.add_argument("--patch_y", type=int, default=2)
    parser.add_argument("--RNNsymmetry", type=str, default="c4v", choices=["nosym", "c4v"])
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--repeats", type=int, default=10, help="Number of timed steady-state steps.")
    parser.add_argument("--warmup_steps", type=int, default=3, help="Total warmup steps; first includes compile.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--jax_cache_dir", type=str, default="jax_cache")
    parser.add_argument("--sanity_samples", type=int, default=16)
    parser.add_argument("--skip_sanity_checks", action="store_true")
    parser.add_argument("--report_memory", action="store_true",
                        help="Report process peak RSS snapshots/deltas in CSV and logs.")
    parser.add_argument("--report_gpu_memory", action="store_true",
                        help="Report GPU memory snapshots via nvidia-smi (if available).")
    parser.add_argument("--print_each_step", action="store_true",
                        help="Print timing for every warmup and timed step.")
    parser.add_argument("--out_csv", type=str, default="trainstep_timing_vs_L.csv")
    parser.add_argument("--out_plot_pdf", type=str, default="trainstep_timing_vs_L.pdf")
    args = parser.parse_args()

    l_values = parse_l_values(args.L_values)
    optimizer = optax.adam(learning_rate=args.lr)
    os.makedirs(args.results_dir, exist_ok=True)
    cache_dir = args.jax_cache_dir if os.path.isabs(args.jax_cache_dir) else os.path.join(args.results_dir, args.jax_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(cache_dir)
    except Exception:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    print(f"JAX compilation cache dir: {cache_dir}")

    out_csv = args.out_csv if os.path.isabs(args.out_csv) else os.path.join(args.results_dir, args.out_csv)
    out_plot_pdf = args.out_plot_pdf if os.path.isabs(args.out_plot_pdf) else os.path.join(args.results_dir, args.out_plot_pdf)

    rows = []

    for l_size in l_values:
        if args.geometry == "square":
            nx = l_size
            ny = l_size
        else:
            nx = args.cylinder_circumference
            ny = l_size

        px_eff, py_eff = effective_patch_dims(nx, ny, args.patch_x, args.patch_y)
        model = TwoDFastGRU(
            d_hidden=args.dh,
            d_model=args.dmodel,
            n_layers=args.num_layers,
            patch_x=px_eff,
            patch_y=py_eff,
        )

        init_key, rng_key = jax.random.split(jax.random.key(1))
        x = jax.random.randint(rng_key, (args.numsamples, nx, ny), 0, 2)
        params = model.init(init_key, x)

        if (not args.skip_sanity_checks) and l_size == l_values[0]:
            sanity_samples = min(args.sanity_samples, args.numsamples)
            run_sanity_checks(model, params, (nx, ny), sanity_samples)

        # Forward-only benchmark on fixed inputs (no sampling/local-energy/optimizer).
        key_fwd = jax.random.key(777 + int(l_size))
        x_fwd = jax.random.randint(key_fwd, (args.numsamples, nx, ny), 0, 2)
        fwd_mode_order = ["parallel", "sequential"] if (l_size % 2 == 0) else ["sequential", "parallel"]
        fwd_stats_by_mode = {}
        for mode in fwd_mode_order:
            fwd_stats_by_mode[mode] = benchmark_forward_only_for_mode(
                model=model,
                params=params,
                samples=x_fwd,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                mode=mode,
                print_each_step=args.print_each_step,
                l_size_for_log=l_size,
            )
        fwd_parallel_stats = fwd_stats_by_mode["parallel"]
        fwd_sequential_stats = fwd_stats_by_mode["sequential"]
        fwd_speedup = (
            fwd_sequential_stats["mean_s"] / fwd_parallel_stats["mean_s"]
            if fwd_parallel_stats["mean_s"] > 0
            else np.nan
        )

        # Alternate order per L to reduce order/caching bias.
        mode_order = ["parallel", "sequential"] if (l_size % 2 == 0) else ["sequential", "parallel"]
        stats_by_mode = {}
        for mode in mode_order:
            stats_by_mode[mode] = benchmark_for_mode(
                model=model,
                params=params,
                optimizer=optimizer,
                numsamples=args.numsamples,
                nx=nx,
                ny=ny,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                mode=mode,
                symmetry=args.RNNsymmetry,
                print_each_step=args.print_each_step,
                l_size_for_log=l_size,
                report_gpu_memory=args.report_gpu_memory,
            )
        parallel_stats = stats_by_mode["parallel"]
        sequential_stats = stats_by_mode["sequential"]

        speedup = sequential_stats["mean_s"] / parallel_stats["mean_s"] if parallel_stats["mean_s"] > 0 else np.nan
        rows.append(
            {
                "L": l_size,
                "Nx": nx,
                "Ny": ny,
                "geometry": args.geometry,
                "symmetry": args.RNNsymmetry,
                "patch_x_eff": px_eff,
                "patch_y_eff": py_eff,
                "parallel_mean_s": parallel_stats["mean_s"],
                "parallel_median_s": parallel_stats["median_s"],
                "parallel_p90_s": parallel_stats["p90_s"],
                "parallel_std_s": parallel_stats["std_s"],
                "parallel_compile_first_step_s": parallel_stats["compile_first_step_s"],
                "parallel_warmup_extra_s": parallel_stats["warmup_extra_s"],
                "parallel_timed_block_s": parallel_stats["timed_block_s"],
                "parallel_peak_rss_before_mb": parallel_stats["peak_rss_before_mb"],
                "parallel_peak_rss_after_compile_mb": parallel_stats["peak_rss_after_compile_mb"],
                "parallel_peak_rss_after_timed_mb": parallel_stats["peak_rss_after_timed_mb"],
                "parallel_delta_peak_rss_compile_mb": parallel_stats["delta_peak_rss_compile_mb"],
                "parallel_delta_peak_rss_timed_mb": parallel_stats["delta_peak_rss_timed_mb"],
                "sequential_mean_s": sequential_stats["mean_s"],
                "sequential_median_s": sequential_stats["median_s"],
                "sequential_p90_s": sequential_stats["p90_s"],
                "sequential_std_s": sequential_stats["std_s"],
                "sequential_compile_first_step_s": sequential_stats["compile_first_step_s"],
                "sequential_warmup_extra_s": sequential_stats["warmup_extra_s"],
                "sequential_timed_block_s": sequential_stats["timed_block_s"],
                "sequential_peak_rss_before_mb": sequential_stats["peak_rss_before_mb"],
                "sequential_peak_rss_after_compile_mb": sequential_stats["peak_rss_after_compile_mb"],
                "sequential_peak_rss_after_timed_mb": sequential_stats["peak_rss_after_timed_mb"],
                "sequential_delta_peak_rss_compile_mb": sequential_stats["delta_peak_rss_compile_mb"],
                "sequential_delta_peak_rss_timed_mb": sequential_stats["delta_peak_rss_timed_mb"],
                "sequential_over_parallel_speedup": speedup,
                "fwd_parallel_mean_s": fwd_parallel_stats["mean_s"],
                "fwd_parallel_median_s": fwd_parallel_stats["median_s"],
                "fwd_parallel_p90_s": fwd_parallel_stats["p90_s"],
                "fwd_parallel_std_s": fwd_parallel_stats["std_s"],
                "fwd_parallel_compile_first_step_s": fwd_parallel_stats["compile_first_step_s"],
                "fwd_parallel_warmup_extra_s": fwd_parallel_stats["warmup_extra_s"],
                "fwd_parallel_timed_block_s": fwd_parallel_stats["timed_block_s"],
                "fwd_sequential_mean_s": fwd_sequential_stats["mean_s"],
                "fwd_sequential_median_s": fwd_sequential_stats["median_s"],
                "fwd_sequential_p90_s": fwd_sequential_stats["p90_s"],
                "fwd_sequential_std_s": fwd_sequential_stats["std_s"],
                "fwd_sequential_compile_first_step_s": fwd_sequential_stats["compile_first_step_s"],
                "fwd_sequential_warmup_extra_s": fwd_sequential_stats["warmup_extra_s"],
                "fwd_sequential_timed_block_s": fwd_sequential_stats["timed_block_s"],
                "fwd_sequential_over_parallel_speedup": fwd_speedup,
                "mode_order": "-".join(mode_order),
            }
        )
        if args.report_gpu_memory:
            rows[-1].update(
                {
                    "parallel_gpu_mem_before_mb": parallel_stats["gpu_mem_before_mb"],
                    "parallel_gpu_mem_after_compile_mb": parallel_stats["gpu_mem_after_compile_mb"],
                    "parallel_gpu_mem_after_timed_mb": parallel_stats["gpu_mem_after_timed_mb"],
                    "parallel_gpu_proc_mem_before_mb": parallel_stats["gpu_proc_mem_before_mb"],
                    "parallel_gpu_proc_mem_after_compile_mb": parallel_stats["gpu_proc_mem_after_compile_mb"],
                    "parallel_gpu_proc_mem_after_timed_mb": parallel_stats["gpu_proc_mem_after_timed_mb"],
                    "parallel_gpu_delta_mem_compile_mb": parallel_stats["gpu_delta_mem_compile_mb"],
                    "parallel_gpu_delta_mem_timed_mb": parallel_stats["gpu_delta_mem_timed_mb"],
                    "sequential_gpu_mem_before_mb": sequential_stats["gpu_mem_before_mb"],
                    "sequential_gpu_mem_after_compile_mb": sequential_stats["gpu_mem_after_compile_mb"],
                    "sequential_gpu_mem_after_timed_mb": sequential_stats["gpu_mem_after_timed_mb"],
                    "sequential_gpu_proc_mem_before_mb": sequential_stats["gpu_proc_mem_before_mb"],
                    "sequential_gpu_proc_mem_after_compile_mb": sequential_stats["gpu_proc_mem_after_compile_mb"],
                    "sequential_gpu_proc_mem_after_timed_mb": sequential_stats["gpu_proc_mem_after_timed_mb"],
                    "sequential_gpu_delta_mem_compile_mb": sequential_stats["gpu_delta_mem_compile_mb"],
                    "sequential_gpu_delta_mem_timed_mb": sequential_stats["gpu_delta_mem_timed_mb"],
                }
            )

        print(
            f"L={l_size} (Nx={nx}, Ny={ny}, geom={args.geometry}): "
            f"sym={args.RNNsymmetry}, patch=({px_eff},{py_eff}) | "
            f"fwd parallel mean/median={fwd_parallel_stats['mean_s']:.6f}/{fwd_parallel_stats['median_s']:.6f}s "
            f"(compile1={fwd_parallel_stats['compile_first_step_s']:.2f}s), "
            f"fwd sequential mean/median={fwd_sequential_stats['mean_s']:.6f}/{fwd_sequential_stats['median_s']:.6f}s "
            f"(compile1={fwd_sequential_stats['compile_first_step_s']:.2f}s), "
            f"fwd speedup(seq/par)={fwd_speedup:.3f}x | "
            f"parallel mean/median={parallel_stats['mean_s']:.6f}/{parallel_stats['median_s']:.6f}s "
            f"(compile1={parallel_stats['compile_first_step_s']:.2f}s), "
            f"sequential mean/median={sequential_stats['mean_s']:.6f}/{sequential_stats['median_s']:.6f}s "
            f"(compile1={sequential_stats['compile_first_step_s']:.2f}s), "
            f"speedup(seq/par)={speedup:.3f}x"
        )
        if args.report_memory:
            print(
                f"  RSS peak MB | parallel: before={parallel_stats['peak_rss_before_mb']:.1f}, "
                f"after_compile={parallel_stats['peak_rss_after_compile_mb']:.1f}, "
                f"after_timed={parallel_stats['peak_rss_after_timed_mb']:.1f}; "
                f"sequential: before={sequential_stats['peak_rss_before_mb']:.1f}, "
                f"after_compile={sequential_stats['peak_rss_after_compile_mb']:.1f}, "
                f"after_timed={sequential_stats['peak_rss_after_timed_mb']:.1f}"
            )
        if args.report_gpu_memory:
            print(
                f"  GPU used MB | parallel: before={parallel_stats['gpu_mem_before_mb']}, "
                f"after_compile={parallel_stats['gpu_mem_after_compile_mb']}, "
                f"after_timed={parallel_stats['gpu_mem_after_timed_mb']}; "
                f"sequential: before={sequential_stats['gpu_mem_before_mb']}, "
                f"after_compile={sequential_stats['gpu_mem_after_compile_mb']}, "
                f"after_timed={sequential_stats['gpu_mem_after_timed_mb']}"
            )
            print(
                f"  GPU proc MB | parallel: before={parallel_stats['gpu_proc_mem_before_mb']}, "
                f"after_compile={parallel_stats['gpu_proc_mem_after_compile_mb']}, "
                f"after_timed={parallel_stats['gpu_proc_mem_after_timed_mb']}; "
                f"sequential: before={sequential_stats['gpu_proc_mem_before_mb']}, "
                f"after_compile={sequential_stats['gpu_proc_mem_after_compile_mb']}, "
                f"after_timed={sequential_stats['gpu_proc_mem_after_timed_mb']}"
            )

    df = pd.DataFrame(rows).sort_values("L")
    df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")
    try:
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "legend.fontsize": 10,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
            }
        )

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.errorbar(
            df["L"],
            df["parallel_mean_s"],
            yerr=df["parallel_std_s"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            color="#1f77b4",
            label="Parallel call",
        )
        ax.errorbar(
            df["L"],
            df["sequential_mean_s"],
            yerr=df["sequential_std_s"],
            marker="s",
            markersize=4,
            linewidth=1.8,
            capsize=3,
            color="#d62728",
            label="Sequential call",
        )
        ax.set_xlabel("System size L")
        ax.set_ylabel("Train-step time (s)")
        ax.set_title("One-step training time vs lattice size")
        ax.grid(True, which="major", linestyle="--", alpha=0.35, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_plot_pdf, bbox_inches="tight")
        print(f"Saved plot: {out_plot_pdf}")
    except ModuleNotFoundError:
        print("matplotlib not found; skipped plot generation.")


if __name__ == "__main__":
    main()
