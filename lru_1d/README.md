# 1D LRU Reproducibility Resources

This folder contains a self-contained implementation for reproducing 1D LRU variational Monte Carlo runs.

## Environment

Create or activate a Python environment with JAX, Flax, Optax, Orbax, NumPy, and Pandas.

If installing from scratch, use:

```bash
pip install -r requirements.txt
```

For GPUs, install the JAX build appropriate for the local CUDA version if the default installation does not detect the GPU.

## Running Training Script

Run the iterative retraining simulation/campaign for the 1D TFIM at h=1, run:

```bash
python run_training.py \
    --L_values "6,8,10,12,16,24,32,48,64,96,128,192,256" \
    --num_layers 3 --seed 1 \
    --iterative-retrain \
    --numsamples 1024 \  # number of samples with which to estimate the energy gradient
    --dh 64 --dmodel 64 --lr 0.0001 \
    -C 40 -F 15 -r 0.25 -s 1.0 \   # exponential decay curve parameters to set number of checkpoints
    --steps-per-checkpoint 1000 \  # number of training steps to perform per checkpoint
    --results_dir "./results/"
```

To run training for each system size sequentially but *without* iterative retraining, simply remove the `--iterative-retrain` flag.

#### Notes/Caveats
The list of `L_values` is sorted in ascending order before any training begins.
The `L0` parameter in the function which computes the number of total training checkpoints to run will be set to `min(L_values)`.

Propagation of model states along during iterative retraining is entirely dependent on the `L_values` that are passed. In the above example, when beginning training for `L=10`, the script will *only* look for a model state for `L=8` to initialize from, and if no such state exists (or the read failed) the script will start training `L=10` from scratch. It will not look for model states for any smaller systems.

Iterative retraining will also reload optimizer states (such as the first and second moment estimates in Adam) from the previous system size in order to avoid having to re-estimate these quantities from scratch for the new system size.


## Expected Output Files

A successful run writes files such as:

```text
seed*/Lx*/h=1.0/lowest_energy/              # orbax checkpoints of the model states with the best energies seen
seed*/Lx*/h=1.0/state/                      # orbax checkpoints of the model states saved periodically
seed*/Lx*/h=1.0/metric_history/batch*.csv   # energy + loss statistics of each training step, for each checkpoint
seed*/Lx*/h=1.0/final_stats.csv             
```

Checkpoints contain the model state, optimizer state, RNG state, as well as some energy statistics (though less thorough that the statistics in the metric_history folder). **NOTE** Due to the how optax implements optimizers, and optimizer *parameters* (such as the learning rate, momentum coefficients, etc.) are *not* saved in the checkpoint, only the actual optimizer *state* is; be careful not to accidentally change the learning rate between runs without realizing.

The `final_stats.csv` file contains estimates of the energy density and standard error at the end of training for a given size, estimated from 1000 x `numsamples` random samples; additionally, we also compute the exact energy density for the system size along with the relative error of the estimated energy density.
If the `final_stats.csv` file is present for a given system size + seed combination, the script will assume the training is complete for that case and move on to the next system size. This should be kept in mind in case one decides to increase the number of training iterations after the fact.
