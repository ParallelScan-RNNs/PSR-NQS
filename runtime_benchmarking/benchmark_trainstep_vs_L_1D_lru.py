import argparse
import os
import time
import resource
import sys
import subprocess

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
import pandas as pd

from ..lru_1d import Positive1DSSMWavefunction
from ..lru_1d.vmc_utils import train_step

sim_dtype = jnp.float32


def parse_l_values(raw: str):
    return sorted([int(x.strip()) for x in raw.split(",") if x.strip()])


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



def benchmark_for_mode(
    model,
    tfim_J,
    tfim_h,
    optimizer,
    numsamples,
    nx,
    repeats,
    warmup_steps,
    parallel,
    print_each_step=False,
    l_size_for_log=None,
    report_gpu_memory=False,
):
    rss_before_mb = peak_rss_mb()
    gpu_before = gpu_memory_snapshot_mb() if report_gpu_memory else None
    
    rngs = nnx.Rngs(0)
    
    site_coords = jnp.array([idx for idx, _ in model.generate_path((nx,))])

    # First step includes JIT compile + first execution.
    t0 = time.perf_counter()
    model, optimizer, _, loss = train_step(model, optimizer, rngs, tfim_J, tfim_h, site_coords, numsamples, nx, parallel, clip_rho=100.0)
    loss.block_until_ready()
    compile_first_step_s = time.perf_counter() - t0
    if print_each_step:
        print(
            f"[step] L={l_size_for_log} parallel={parallel} phase=compile_first "
            f"step=1/{max(1, warmup_steps)} dt={compile_first_step_s:.6f}s"
        )
    rss_after_compile_mb = peak_rss_mb()
    gpu_after_compile = gpu_memory_snapshot_mb() if report_gpu_memory else None


    warmup_extra_start = time.perf_counter()
    for i in range(max(0, warmup_steps - 1)):
        wi0 = time.perf_counter()
        model, optimizer, _, loss = train_step(model, optimizer, rngs, tfim_J, tfim_h, site_coords, numsamples, nx, parallel, clip_rho=100.0)
        loss.block_until_ready()
        dt = time.perf_counter() - wi0
        if print_each_step:
            print(
                f"[step] L={l_size_for_log} parallel={parallel} phase=warmup "
                f"step={i + 2}/{warmup_steps} dt={dt:.6f}s"
            )
    warmup_extra_s = time.perf_counter() - warmup_extra_start
    rss_after_warmup_mb = peak_rss_mb()

    timed_block_start = time.perf_counter()
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        model, optimizer, _, loss = train_step(model, optimizer, rngs, tfim_J, tfim_h, site_coords, numsamples, nx, parallel, clip_rho=100.0)
        loss.block_until_ready()
        dt = time.perf_counter() - start
        times.append(dt)
        if print_each_step:
            print(
                f"[step] L={l_size_for_log} parallel={parallel} phase=timed "
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
    samples,
    repeats,
    warmup_steps,
    parallel,
    print_each_step=False,
    l_size_for_log=None,
):
    @nnx.jit
    def fwd_fn(model, samples):
        return model.log_probs(samples, parallel=parallel)

    # First call includes compile + first execution.
    t0 = time.perf_counter()
    log_probs = fwd_fn(model, samples)
    log_probs.block_until_ready()
    compile_first_step_s = time.perf_counter() - t0
    if print_each_step:
        print(
            f"[fwd] L={l_size_for_log} parallel={parallel} phase=compile_first "
            f"step=1/{max(1, warmup_steps)} dt={compile_first_step_s:.6f}s"
        )

    warmup_extra_start = time.perf_counter()
    for i in range(max(0, warmup_steps - 1)):
        wi0 = time.perf_counter()
        log_probs = fwd_fn(model, samples)
        log_probs.block_until_ready()
        if print_each_step:
            print(
                f"[fwd] L={l_size_for_log} parallel={parallel} phase=warmup "
                f"step={i + 2}/{warmup_steps} dt={time.perf_counter() - wi0:.6f}s"
            )
    warmup_extra_s = time.perf_counter() - warmup_extra_start

    timed_block_start = time.perf_counter()
    times = []
    for i in range(repeats):
        start = time.perf_counter()
        log_probs = fwd_fn(model, samples)
        log_probs.block_until_ready()
        dt = time.perf_counter() - start
        times.append(dt)
        if print_each_step:
            print(
                f"[fwd] L={l_size_for_log} parallel={parallel} phase=timed "
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


def run_sanity_checks(model, l_size, num_sanity_samples):
    rngs = nnx.Rngs(123)
    x_rand = rngs.randint(shape=(num_sanity_samples, l_size[0]), minval=0, maxval=2)

    logp_parallel = model.log_probs(x_rand, parallel=True)
    logp_sequential = model.log_probs(x_rand, parallel=False)
    diff_rand = jnp.abs(logp_parallel - logp_sequential)
    print(
        "[Sanity] random-input logp |parallel-sequential|: "
        f"max={float(jnp.max(diff_rand)):.3e}, mean={float(jnp.mean(diff_rand)):.3e}"
    )

    rngs = nnx.Rngs(456)
    samples, clp = model.sample(l_size, num_sanity_samples, rngs=rngs)
    logp_clp = model.log_probs(samples, clp)  # parallel doesnt do anything when passing the conditionals
    logp_sample_parallel =  model.log_probs(samples, clp, parallel=True)
    logp_sample_sequential =  model.log_probs(samples, clp, parallel=False)
    diff_sample = jnp.abs(logp_sample_parallel - logp_sample_sequential)
    print(
        "[Sanity] sampled-input logp |parallel-sequential|: "
        f"max={float(jnp.max(diff_sample)):.3e}, mean={float(jnp.mean(diff_sample)):.3e}"
    )
    diff_sample = jnp.abs(logp_clp - logp_sample_sequential)
    print(
        "[Sanity] sampled-input logp |clp-sequential|: "
        f"max={float(jnp.max(diff_sample)):.3e}, mean={float(jnp.mean(diff_sample)):.3e}"
    )

    sample_a, clp_a = model.sample(l_size, num_sanity_samples, rngs=nnx.Rngs(789))
    sample_b, clp_b = model.sample(l_size, num_sanity_samples, rngs=nnx.Rngs(789))
    deterministic = bool(jnp.array_equal(sample_a, sample_b))
    print(f"[Sanity] sample deterministic for same key: {deterministic}")
    deterministic = bool(jnp.array_equal(clp_a, clp_b))
    print(f"[Sanity] cond_log_psis deterministic for same key: {deterministic}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark one train-step time vs L for parallel vs sequential calls")
    parser.add_argument("--L_values", type=str, default="8,16,32,64,128,256,512")
    parser.add_argument("--unroll", action="store_true")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dh", type=int, default=256)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--numsamples", type=int, default=100)
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
    print(l_values)
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
        nx = l_size

        model = Positive1DSSMWavefunction(
            local_dim=2,
            ssm_dim=args.dh,
            emb_dim=args.dmodel,
            num_layers=args.num_layers,
            rngs=nnx.Rngs(1),
            unroll=args.unroll,
        )
        
        opt = nnx.Optimizer(model, optimizer, wrt=nnx.Param)
        print(f"Benchmarking nx={nx}...")
        if (not args.skip_sanity_checks) and l_size == l_values[0]:
            sanity_samples = min(args.sanity_samples, args.numsamples)
            run_sanity_checks(model, (nx,), sanity_samples)

        # Forward-only benchmark on fixed inputs (no sampling/local-energy/optimizer).
        rng_fwd = nnx.Rngs(777 + int(l_size))
        x_fwd = rng_fwd.randint((args.numsamples, nx), 0, 2)
        fwd_mode_order = ["parallel", "sequential"] if (l_size % 2 == 0) else ["sequential", "parallel"]
        fwd_stats_by_mode = {}
        for mode in fwd_mode_order:
            fwd_stats_by_mode[mode] = benchmark_forward_only_for_mode(
                model=model,
                samples=x_fwd,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                parallel=(mode == "parallel"),
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
                tfim_J=-1.0,
                tfim_h=1.0,
                optimizer=opt,
                numsamples=args.numsamples,
                nx=nx,
                repeats=args.repeats,
                warmup_steps=args.warmup_steps,
                parallel=(mode == "parallel"),
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
            f"L={l_size} (Nx={nx},): "
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
            
        # save to csv every time we finish a size
        df = pd.DataFrame(rows).sort_values("L")
        df.to_csv(out_csv, index=False, mode='w')  # overwrite
        print(f"Saved CSV: {out_csv}")
        
    df = pd.DataFrame(rows).sort_values("L")

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
