# System Size Runtime Scaling Reproducibility Scripts

This folder contains a self-contained implementation for reproducing the 1D LRU and 2D minGRU variational Monte Carlo training step runtime scaling plots. 
The 1D LRU benchmarks estimate the energy for a 1D TFIM at the QCP (h=1), while the 2D minGRU benchmarks estimate the energy for a 2D AFM Heisenberg model. Both assume open boundary conditions.


## Environment

Create or activate a Python environment with JAX, Flax, Optax, NumPy, and Pandas.

If installing from scratch, use:

```bash
pip install -r requirements.txt
```

For GPUs, install the JAX build appropriate for the local CUDA version if the default installation does not detect the GPU.

## Running the benchmarks

To produce the data for the 1D LRU benchmarks, along with some simple scaling plots, run:
```bash
python benchmark_trainstep_vs_L_1D_lru.py \
    --L_values "6,8,10,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536" \
    --num_layers 1 --repeats 100 \
    --numsamples 128 --dh 32 --dmodel 32 \
    --report_memory --report_gpu_memory \
    --results_dir "./results/" \
    --out_csv "trainstep_timing_vs_L_1D_lru.csv" \
    --out_plot_pdf "trainstep_timing_vs_L_1D_lru.pdf"
```
Although we recommend breaking up the run into a few smaller groups of L values.

To produce the data for the 2D minGRU benchmarks, along with some simple scaling plots, run:
```bash
python benchmark_trainstep_vs_L_2D_minGRU.py \
    --L_values "6,8,10,12,16,24,32" \
    --num_layers 1 \
    --numsamples 8 \
    --report_memory --report_gpu_memory \
    --results_dir "./results/" \
    --out_csv "trainstep_timing_vs_L_gru.csv" \
    --out_plot_pdf "trainstep_timing_vs_L_gru.pdf"
```
Note that the minGRU uses 2x2 patching by default, so `L_values` must all be even.
