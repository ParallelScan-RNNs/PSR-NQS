import argparse
import os
import time
from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import orbax.checkpoint as ocp

import pandas as pd

from .wavefunctions import Positive1DSSMWavefunction
from .vmc_utils import compute_local_energies_scan, train_step


def parse_l_values(raw: str):
    return sorted([int(x.strip()) for x in raw.split(",") if x.strip()])

def exact_obc_critical_energy_density(N):
    return -(1/jnp.sin(jnp.pi/(2*(2*N + 1))) - 1)/N

def exact_pbc_critical_energy_density(N):
    return -(2/jnp.sin(jnp.pi/(2*N)))/N


def train(
    model,
    tfim_J,
    tfim_h,
    optimizer,
    numsamples,
    nx,
    num_checkpoints,
    steps_per_checkpoint,
    parallel,
    clip_rho,
    rngs,
    results_dir,
    ckpt_dir,
    best_ckpt_dir,
    prev_ckpt_dir
):
    # --- Checkpoint manager for rolling checkpoints ---
    ckpt_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=10, create=True, cleanup_tmp_directories=True),
    )
    
    # --- Separate checkpointer for best energy_mean model ---
    best_ckpt_manager = ocp.CheckpointManager(
        best_ckpt_dir,
        options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True, cleanup_tmp_directories=True),
    )
    best_energy_mean = float("inf")
    best_energy_mean_error = 1.0

    # --- Resume logic ---
    starting_epoch = 1
    print(f"\tLooking for checkpoints in {ckpt_dir} to resume from...")
    if ckpt_manager.latest_step() is not None:
        # True resume from ckpt_dir
        last_step = ckpt_manager.latest_step()
        print(f"\t\tResuming from checkpoint at step {last_step}")

        restored = ckpt_manager.restore(last_step, args=ocp.args.StandardRestore({
            "model": nnx.split(model)[1],
            "optimizer": nnx.split(optimizer)[1],
            "rngs": nnx.split(rngs)[1],
            "best_energy_density_mean": best_energy_mean,
        })) # type: ignore
        nnx.update(model, restored["model"])
        nnx.update(optimizer, restored["optimizer"])
        nnx.update(rngs, restored["rngs"])
        best_energy_mean = float(restored["best_energy_density_mean"]) * nx
        if ckpt_manager.metrics(last_step) is not None and "best_energy_density_error_of_mean" in ckpt_manager.metrics(last_step).keys():
            best_energy_mean_error = float(ckpt_manager.metrics(last_step).get("best_energy_density_error_of_mean")) * nx
        else:
            # in case the checkpoint has no metric info, we just set the error to 1.0 so that any new energy_mean 
            #  will be considered an improvement and trigger a checkpoint save to best_ckpt_manager
            # this makes sense bc error is usually on the order of 0.01 or smaller, so if we have no info on the error,
            #  we should be optimistic and just assume any new energy_mean is better than the old one to encourage 
            #  finding a good checkpoint to save as best_ckpt early on in training.
            best_energy_mean_error = 1.0

        starting_epoch = last_step + 1  # type: ignore
        print(f"\t\t\tStarting from checkpoint {starting_epoch}, best_energy_mean so far: {best_energy_mean:.6f} +/- {best_energy_mean_error:.6f}.")

    elif prev_ckpt_dir:
        # Warm start: load weights from prev_ckpt_dir
        prev_manager = ocp.CheckpointManager(prev_ckpt_dir)
        if prev_manager.latest_step() is not None:
            last_step = prev_manager.latest_step()
            print(f"\t\tNo checkpoints in ckpt_dir ({ckpt_dir}). Loading pretrained weights from prev_ckpt_dir ({prev_ckpt_dir}) (step {last_step})")
            
            restored = prev_manager.restore(last_step, args=ocp.args.StandardRestore({
                "model": nnx.split(model)[1],
                "optimizer": nnx.split(optimizer)[1],
                "rngs": nnx.split(rngs)[1],
                "best_energy_density_mean": 0.0,
            })) # type: ignore
            nnx.update(model, restored["model"])
            nnx.update(optimizer, restored["optimizer"])
            nnx.update(rngs, restored["rngs"])
            best_energy_mean = float(restored["best_energy_density_mean"]) * nx
            if prev_manager.metrics(last_step) is not None and "best_energy_density_error_of_mean" in prev_manager.metrics(last_step).keys():
                best_energy_mean_error = float(prev_manager.metrics(last_step).get("best_energy_density_error_of_mean")) * nx
            else:
                # in case the checkpoint has no metric info, we just set the error to 1.0 so that any new energy_mean 
                #  will be considered an improvement and trigger a checkpoint save to best_ckpt_manager
                # this makes sense bc error is usually on the order of 0.01 or smaller, so if we have no info on the error,
                #  we should be optimistic and just assume any new energy_mean is better than the old one to encourage 
                #  finding a good checkpoint to save as best_ckpt early on in training.
                best_energy_mean_error = 1.0
            
            print(f"\t\t\tStarting from epoch 1 with loaded weights, best_energy_mean so far: {best_energy_mean:.6f} +/- {best_energy_mean_error:.6f}.")

        else:
            print(f"\t\tNo checkpoints found in prev_ckpt_dir ({prev_ckpt_dir}), starting from scratch.")
    else:
        print("\t\tNo checkpoint found, starting from scratch.")

    site_coords = jnp.array([idx for idx, _ in model.generate_path((nx,))])

    # --- Training loop ---
    stats_history = []
    checkpoint_start_time = time.perf_counter()
    print(f"\n\tStarting training for {num_checkpoints*steps_per_checkpoint} iterations, at epoch {starting_epoch}, with checkpoints every {steps_per_checkpoint} iterations.", flush=True)
    for epoch in range(starting_epoch, num_checkpoints*steps_per_checkpoint + 1):
        model, optimizer, stats, loss = train_step(model, optimizer, rngs, tfim_J, tfim_h, site_coords, numsamples, nx, parallel, clip_rho)
        loss.block_until_ready()
        
        stats = {k: float(v) for k,v in stats.items()}
        stats_history.append(stats)
        
        # only start considering saving to best_ckpt_manager after we've done at least half of the checkpoints, 
        # to give the model a chance to reduce its variance enough to get a meaningful estimate of the energy_mean and energy_error_of_mean 
        # before we start using those metrics to decide whether to save checkpoints to best_ckpt_manager or not.
        if epoch > (num_checkpoints*steps_per_checkpoint // 2):
            # compare Upper Confidence Bound of current energy_mean with best_energy_mean seen so far to 
            #  decide whether to save checkpoint to best_ckpt_manager
            n_sigma = 1.0
            if stats["energy_mean"] + n_sigma * stats["energy_error_of_mean"] < best_energy_mean + n_sigma * best_energy_mean_error:
                best_energy_mean = stats["energy_mean"]
                best_energy_mean_error = stats["energy_error_of_mean"]
                print(f"\t\t\tNew best energy_mean {best_energy_mean:.6f} +/- {best_energy_mean_error:.6f} at checkpoint {epoch}.", end=" ")

                best_ckpt_manager.save(epoch, args=ocp.args.StandardSave({
                    "model": nnx.split(model)[1],
                    "optimizer": nnx.split(optimizer)[1],
                    "rngs": nnx.split(rngs)[1],
                    "best_energy_density_mean": best_energy_mean / nx,  #also save the best energy seen so far just so we can consistently track it across checkpoints
                }), metrics={ # type: ignore
                    "stats": stats, 
                    "best_energy_density_mean": best_energy_mean / nx, 
                    "best_energy_density_error_of_mean": best_energy_mean_error / nx
                })
                print(f"Saved checkpoint {epoch}.")


        if epoch % steps_per_checkpoint == 0:
            print(f"\t\tStep {epoch}: loss = {loss}, energy = {stats['energy_mean']:.6f} +/- {stats['energy_error_of_mean']:.6f}.", end=" ")
            
            # Save rolling checkpoint
            ckpt_manager.save(epoch, args=ocp.args.StandardSave({
                "model": nnx.split(model)[1],
                "optimizer": nnx.split(optimizer)[1],
                "rngs": nnx.split(rngs)[1],
                "best_energy_density_mean": best_energy_mean / nx,  #also save the best energy seen so far just so we can consistently track it across checkpoints
            }), metrics={ # type: ignore
                "stats": stats, 
                "best_energy_density_mean": best_energy_mean / nx, 
                "best_energy_density_error_of_mean": best_energy_mean_error / nx
            })
            print(f"Saved checkpoint {epoch}.", end=" ")
            
            out_csv = os.path.join(results_dir, "metric_history", f"batch{(epoch - 1) // steps_per_checkpoint}.csv")
            history = pd.DataFrame.from_records(stats_history)
            history.to_csv(out_csv, index=False)
            
            stats_history = []
            
            elapsed_time = time.perf_counter() - checkpoint_start_time
            print(f"{steps_per_checkpoint} iterations took: {timedelta(seconds=elapsed_time)}", flush=True)
            checkpoint_start_time = time.perf_counter()
                
    print("\n\tTraining complete, now estimating final energy with more samples...", flush=True)
    n_iter = 1_000
    n_samples = numsamples
    
    sampling_start_time = time.perf_counter()
    @nnx.scan(in_axes=(None, None, None, nnx.Carry),
              out_axes=nnx.Carry,
              length=n_iter)
    def scan_final_energy(model, rngs, site_coords, carry):
        full_mean, full_M2, i = carry
        
        samples, clp = model.sample((nx,), n_samples, rngs=rngs)
        log_psi_original = model.log_psis(samples, clp)
        
        E_loc = compute_local_energies_scan(model, tfim_J, tfim_h, site_coords, samples, log_psi_original, parallel=parallel) / nx
        E_mean = jnp.mean(E_loc.real)
        E_var = jnp.var(E_loc.real, mean=E_mean)
        
        delta = E_mean - full_mean
        w = delta / (i + 1)
        
        return (full_mean + w, 
                full_M2 + n_samples * (E_var + delta * (i*w)),
                i + 1)
    
    full_mean, full_M2, _ = scan_final_energy(model, rngs, site_coords, (0.0, 0.0, 0))
        
    total_count = n_samples * n_iter
    E_mean = float(full_mean)
    E_var = float(full_M2 / total_count)
    E_err = float(jnp.sqrt(E_var / total_count))
    exact_E = float(exact_obc_critical_energy_density(nx))
    rel_err = abs(E_mean - exact_E) / abs(exact_E)
    final_stats = pd.DataFrame.from_records([{"Lx": nx, "exact_obc": exact_E,
                                              "energy_mean": E_mean, "energy_error_of_mean": E_err,
                                              "rel_err": rel_err,
                                              "energy_var": E_var, "num_samples": int(total_count)}])
    out_csv = os.path.join(results_dir, f"final_stats.csv")
    final_stats.to_csv(out_csv, index=False)
    
    print(f"\t\tFinal energy density = {E_mean} +/- {E_err}  (estimated from {total_count} samples, time taken: {timedelta(seconds=time.perf_counter() - sampling_start_time)})")
    print(f"\t\tExact (OBC): {exact_E}, relative error: {rel_err}", flush=True)
   
    best_ckpt_manager.close()
    ckpt_manager.close()


def num_checkpoints_curve(L, L0=6, s=1.0, C=40, r=0.25, F=15):
    return np.round((C * (np.exp(-r * (L - L0))) + F) * s).astype(int).item()


def main():
    parser = argparse.ArgumentParser(description="Benchmark one train-step time vs L for parallel vs sequential calls")
    parser.add_argument("--L_values", type=str, default="10")
    parser.add_argument("--iterative-retrain", action="store_true", help="If true, will train larger L_values using trained parameters of the previous system size")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dh", type=int, default=256)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--numsamples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--clip-rho", type=float, default=10.0)
    parser.add_argument("-C", type=int, default=40, help="C parameter for the step curve calculation.")
    parser.add_argument("-F", type=int, default=15, help="F parameter for the step curve calculation.")
    parser.add_argument("-r", type=float, default=0.25, help="r parameter for the step curve calculation.")
    parser.add_argument("-s", type=float, default=1.0, help="s parameter for the step curve calculation.")
    parser.add_argument("--steps-per-checkpoint", type=int, default=1000, help="Number of training iterations to do between each checkpoint.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--jax_cache_dir", type=str, default="jax_cache")
    args = parser.parse_args()

    
    os.makedirs(args.results_dir, exist_ok=True)
    cache_dir = args.jax_cache_dir if os.path.isabs(args.jax_cache_dir) else os.path.join(args.results_dir, args.jax_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from jax.experimental.compilation_cache import compilation_cache as cc

        cc.set_cache_dir(cache_dir)
    except Exception:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
    print(f"JAX compilation cache dir: {cache_dir}\n")
    
    ckpt_dir = None
    
    print("\n", "-"*50, "\n", flush=True)
    l_values = parse_l_values(args.L_values)
    L0 = min(l_values)
    print("L_values:", l_values)
    print("num_checkpoints for each L:", [num_checkpoints_curve(L, L0=L0, s=args.s, C=args.C, r=args.r, F=args.F) for L in l_values])
    print("\n", "-"*50, "\n", flush=True)
    
    optimizer = optax.adam(learning_rate=args.lr)
    transverse_field = 1.0
    for nx in l_values:
        start_time = time.perf_counter()
        
        num_checkpoints = num_checkpoints_curve(nx, L0=L0, s=args.s, C=args.C, r=args.r, F=args.F)
        num_samples = args.numsamples
        rngs = nnx.Rngs(args.seed)
        
        prev_ckpt_dir = ckpt_dir
        out_dir = os.path.abspath(os.path.join(args.results_dir, f"seed{args.seed}", f"Lx{nx}", f"h={transverse_field}"))
        os.makedirs(os.path.join(out_dir, "metric_history"), exist_ok=True)
        ckpt_dir = os.path.join(out_dir, "state")
        best_ckpt_dir = os.path.join(out_dir, "lowest_energy")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(best_ckpt_dir, exist_ok=True)
        
        out_csv = os.path.join(out_dir, f"final_stats.csv")
        if os.path.exists(out_csv) and os.path.isfile(out_csv):
            print(f"Final stats already exist at {out_csv}, skipping training for nx={nx}.")
            continue
        
        model = Positive1DSSMWavefunction(
            local_dim=2,
            ssm_dim=args.dh,
            emb_dim=args.dmodel,
            num_layers=args.num_layers,
            rngs=rngs,
            parallel=True,
        )
        
        
        opt = nnx.Optimizer(model, optimizer, wrt=nnx.Param)
        print(f"Training nx={nx}, seed={args.seed}, h={transverse_field}...")
        print("num parameters:", sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))))

        train(
            model=model,
            tfim_J=-1.0, 
            tfim_h=transverse_field,
            optimizer=opt,
            numsamples=num_samples,
            nx=nx,
            num_checkpoints=num_checkpoints,
            steps_per_checkpoint=args.steps_per_checkpoint,
            parallel=True,
            clip_rho=args.clip_rho,
            rngs=rngs,
            results_dir=out_dir,
            ckpt_dir=ckpt_dir,
            best_ckpt_dir=best_ckpt_dir,
            prev_ckpt_dir=(prev_ckpt_dir if args.iterative_retrain else None)
        )
        
        elapsed_time = time.perf_counter() - start_time
        print(f"Training nx={nx}, seed={args.seed}, h={transverse_field} took {timedelta(seconds=elapsed_time)} seconds")
        print("\n", "-"*50, "\n", flush=True)
        

if __name__ == "__main__":
    main()
