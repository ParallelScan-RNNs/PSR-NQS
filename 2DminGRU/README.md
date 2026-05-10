# 2D minGRU Reproducibility Repo

This folder contains a self-contained implementation for reproducing the 2D minGRU variational Monte Carlo runs.

## Files

- `model.py`: 2D minGRU autoregressive wave-function model.
- `Helper_functions.py`: local-energy and utility functions.
- `run_coldstart.py`: cold-start training script. By default it runs the standard cold-start campaign for `L=10` and `L=16`; it can also run a single explicitly specified lattice size.
- `run_iterative.py`: iterative-retraining script from `L=6` to `L=50`.
- `requirements.txt`: Python package requirements.

## Installation

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The code uses JAX, Flax, and Optax. On GPU systems, install the JAX build appropriate for your CUDA version if needed.

## Quick start

From this folder, first run dry runs to print the selected configurations without launching training:

```bash
python3 run_coldstart.py --dry-run
python3 run_iterative.py --dry-run
```

Run the cold-start calculations:

```bash
python3 run_coldstart.py
```

Run the iterative-retraining calculation:

```bash
python3 run_iterative.py
```

## Useful options

For cold-start runs:

```bash
python3 run_coldstart.py --only-L 10
python3 run_coldstart.py --only-L 10,16
python3 run_coldstart.py --num_epochs 1000
```

For a single cold-start training run:

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

For iterative retraining:

```bash
python3 run_iterative.py --l-min 6 --l-max 20
python3 run_iterative.py --base-dir ./TwoDminGRU_iter
python3 run_iterative.py --seed 1
python3 run_iterative.py --no-resume
```

## Output locations

Cold-start outputs are written by default to:

```text
./TwoDminGRU/L_<L>/dh_512/numlayers_6/
```

Iterative-retraining outputs are written by default to:

```text
./TwoDminGRU_iter/L_<L>/dh_256/numlayers_3/
```

Each output directory contains checkpoints, training traces, and final-energy estimates.

## Notes

- Checkpoints are saved during training and are reused by default when rerunning the same configuration.
- Use `--no-resume` with `run_iterative.py` to ignore existing iterative checkpoints.
