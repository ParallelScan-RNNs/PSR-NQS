# 2D minGRU Reproducibility Repo

This folder contains a self-contained implementation for reproducing 2D minGRU variational Monte Carlo runs.

The main entry points are:

- `model.py`: 2D minGRU autoregressive wave-function model.
- `Helper_functions.py`: local-energy and utility functions.
- `run_coldstart.py`: cold-start training script.
- `run_iterative.py`: iterative-retraining script from small to larger lattice sizes.
- `requirements.txt`: Python package requirements.

## Environment

Create or activate a Python environment with JAX, Flax, Optax, NumPy, and pandas.

If installing from scratch, use:

```bash
pip install -r requirements.txt
```

For GPUs, install the JAX build appropriate for the local CUDA version if the default installation does not detect the GPU.

## Quick Verification

Before running expensive jobs, verify syntax and launch configuration:

```bash
python3 -m py_compile run_iterative.py run_coldstart.py model.py Helper_functions.py
python3 run_iterative.py --dry-run
python3 run_coldstart.py --dry-run
```

The iterative dry run should print the selected lattice schedule and planned checkpoint paths. The cold-start dry run should print the selected cold-start commands for `L=10` and `L=16`.

## Numerical Sanity Checks

The following checks are useful for reproducibility and debugging:

- `sample()` returns binary arrays of shape `(numsamples, L, L)`.
- `model.apply(params, samples)` gives finite log-probabilities for sampled states.
- The parallel call `model.apply(params, samples)` agrees with `sequential_call` to float32 tolerance.
- For small systems such as `L=4`, summing `exp(log_prob)` over all states gives approximately `1.0`.

A compact consistency script:

```bash
import itertools
import jax
import jax.numpy as jnp
from model import TwoDFastGRU

L = 4
px = py = 2
numsamples = 16

model = TwoDFastGRU(d_hidden=8, d_model=8, n_layers=1, patch_x=px, patch_y=py)
init_samples = jax.random.randint(jax.random.key(0), (numsamples, L, L), 0, 2)
params = model.init(jax.random.key(1), init_samples)

sampled = model.apply(params, jax.random.key(2), numsamples, L, L, method="sample")
lp_parallel = model.apply(params, sampled)
lp_sequential = model.apply(params, sampled, method="sequential_call")

print("sampled shape", sampled.shape)
print("sample min/max", int(jnp.min(sampled)), int(jnp.max(sampled)))
print("sample logprob finite", bool(jnp.all(jnp.isfinite(lp_parallel))))
print("parallel/sequential max diff", float(jnp.max(jnp.abs(lp_parallel - lp_sequential))))

states = jnp.array(list(itertools.product([0, 1], repeat=L * L)), dtype=jnp.int32).reshape(-1, L, L)
all_lp = model.apply(params, states)
print("normalization", float(jnp.sum(jnp.exp(all_lp))))
```

The max difference should be near zero, up to float32 roundoff, and the normalization should be close to `1.0`.

## C4v Symmetry Convention

The C4v-symmetrized probability is implemented as:

```text
P_sym(x) = (1 / |G|) sum_g P(gx)
log P_sym(x) = logsumexp_g(log P(gx)) - log(|G|)
log psi(x) = 0.5 * log P_sym(x)
```

The eight transformations are identity, rotations by 90/180/270 degrees, row flip, column flip, main-diagonal transpose, and anti-diagonal transpose.

This convention assumes a real nonnegative wave-function amplitude. 

C4v symmetry is intended for square `L x L` lattices. The production schedules in this repo use square lattices.

## Running Cold-Start Calculations

Dry run:

```bash
python3 run_coldstart.py --dry-run
```

Run the default cold-start campaign:

```bash
python3 run_coldstart.py
```

Run only selected lattice sizes:

```bash
python3 run_coldstart.py --only-L 10
python3 run_coldstart.py --only-L 10,16
```

Run a single cold-start training configuration:

```bash
python3 run_coldstart.py \
  --L 10 \
  --num_layers 6 \
  --dh 512 \
  --dmodel 512 \
  --lr 5e-4 \
  --lrdecaytime 5000 \
  --RNNsymmetry c4v \
  --numsamples 200 \
  --num_epochs 150000 \
  --patch_x 2 \
  --patch_y 2
```

Cold-start outputs are written by default to:

```text
./TwoDminGRU/L_<L>/dh_<dh>/numlayers_<num_layers>/
```

## Running Iterative Retraining

Dry run:

```bash
python3 run_iterative.py --dry-run
```

Run the default iterative campaign:

```bash
python3 run_iterative.py
```

Limit the lattice range:

```bash
python3 run_iterative.py --l-min 6 --l-max 20
```

Change the output directory:

```bash
python3 run_iterative.py --base-dir ./TwoDminGRU_iter
```

Use a different seed:

```bash
python3 run_iterative.py --seed 1
```

Ignore existing iterative checkpoints:

```bash
python3 run_iterative.py --no-resume
```

Iterative outputs are written by default to:

```text
./TwoDminGRU_iter/L_<L>/dh_256/numlayers_3/
```

## Expected Output Files

A successful run writes files such as:

```text
Params/checkpoint*.pkl
Outputs/outputs*.csv
Outputs/final_energy*.txt
energies*.txt
variances*.txt
```

Checkpoints contain the model state, optimizer state, RNG key, energy/variance traces, timing information, and metadata.

Iterative checkpoints are reused by default. If a checkpoint is marked complete, rerunning with resume enabled should load it and skip retraining for that lattice size.

## Practical Reproducibility Notes

- Use `--dry-run` before launching long jobs.
- Run the numerical sanity checks after changing `model.py`, `Helper_functions.py`, patch sizes, or symmetry code.
- Keep `dmodel == dh` unless the model skip-connection projection is changed deliberately.
