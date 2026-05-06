# 2D minGRU Reproducibility Repo

This folder is self-contained for reproducing the two sets of runs:

1. Cold start (`L=10,16`)
2. Iterative retraining (`L=6..50` with per-`L` overrides)

All required Python source files are copied locally:

- `run_minGRU_skipconnection.py`
- `run_minGRU_iterative_retraining.py`
- `model_skipconnection.py`
- `Helper_functions.py`

## Quick start

From this folder:

Dry run (print planned commands/configs):

```bash
python3 run_coldstart.py --dry-run
python3 run_iterative.py --dry-run
```

Run cold start:

```bash
python3 run_coldstart.py
```

Run iterative retraining:

```bash
python3 run_iterative.py
```

## Optional flags

Cold start:

- `--only-L 10` (or `10,16`)
- `--num-epochs 1000` (for quick test runs)

Iterative retraining:

- `--l-min 6 --l-max 50` (run subset)
- `--base-dir ./TwoDminGRU_iter`
- `--seed 1`
- `--no-resume'

## Outputs

Cold-start outputs are written by default under:

- `./TwoDminGRU/L_<L>/dh_512/numlayers_6/`

Iterative outputs are written by default under:

- `./TwoDminGRU_iter/L_<L>/dh_256/numlayers_3/`


