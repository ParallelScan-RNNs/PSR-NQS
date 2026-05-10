import jax
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from typing import List, Tuple, Union, Optional, Callable, Any
import optax
import pickle
# from tqdm import tqdm
from functools import partial
from jax import jit
import time
import argparse
import os
import subprocess
import sys
import pandas as pd
from jax import profiler

from Helper_functions import *
from model import *

# sim_dtype = jnp.bfloat16
sim_dtype = jnp.float32

starttime = time.time()

##############
parser = argparse.ArgumentParser(description='2D minGRU training and cold-start launcher')
parser.add_argument('--num_layers', type=int, help='')
parser.add_argument('--L', type=int, help='')
parser.add_argument('--dh', type=int, default=64, help='')
parser.add_argument('--dmodel', type=int, default=64, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--lrdecaytime', type=int, default=5000, help='')
parser.add_argument('--RNNsymmetry', type=str, default="nosym", help='')
parser.add_argument('--numsamples', type=int, default=200, help='')
parser.add_argument('--num_epochs', type=int, default=150000, help='')
parser.add_argument('--patch_x', type=int, default=1, help='')
parser.add_argument('--patch_y', type=int, default=8, help='')
parser.add_argument('--dotraining', type = str2bool, default=True)

# With no explicit --L/--num_layers, this script launches the standard
# cold-start campaign for L=10 and L=16.
parser.add_argument('--dry-run', action='store_true', help='Print cold-start commands without launching training.')
parser.add_argument('--only-L', default='', help='Comma-separated cold-start filter, e.g. 10 or 10,16.')
parser.add_argument('--python-bin', default=sys.executable, help='Python executable used when launching cold-start jobs.')

args = parser.parse_args()

COLDSTART_RUNS = [
    {
        'L': 10,
        'num_layers': 6,
        'dh': 512,
        'dmodel': 512,
        'lr': 5e-4,
        'lrdecaytime': 5000,
        'RNNsymmetry': 'c4v',
        'numsamples': 200,
        'num_epochs': 150000,
        'patch_x': 2,
        'patch_y': 2,
    },
    {
        'L': 16,
        'num_layers': 6,
        'dh': 512,
        'dmodel': 512,
        'lr': 5e-4,
        'lrdecaytime': 5000,
        'RNNsymmetry': 'c4v',
        'numsamples': 200,
        'num_epochs': 150000,
        'patch_x': 2,
        'patch_y': 2,
    },
]

campaign_mode = (args.L is None and args.num_layers is None)

if campaign_mode:
    only = set()
    if args.only_L:
        only = {int(x.strip()) for x in args.only_L.split(',') if x.strip()}

    selected = [cfg.copy() for cfg in COLDSTART_RUNS if not only or cfg['L'] in only]
    if args.num_epochs != parser.get_default('num_epochs'):
        for cfg in selected:
            cfg['num_epochs'] = args.num_epochs

    if not selected:
        raise ValueError('No cold-start runs selected.')

    here = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.abspath(__file__)

    print('[coldstart] selected L:', [cfg['L'] for cfg in selected])
    for cfg in selected:
        cmd = [args.python_bin, runner]
        for k, v in cfg.items():
            cmd.extend([f'--{k}', str(v)])

        print('\n[coldstart]', ' '.join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=here, check=True)

    print('\n[coldstart] done')
    sys.exit(0)

if args.L is None or args.num_layers is None:
    raise ValueError('For a single training run, provide both --L and --num_layers. With no training arguments, the standard cold-start campaign is launched.')

L = args.L
num_layers = args.num_layers
dh = args.dh
dmodel = args.dmodel
RNNsymmetry = args.RNNsymmetry
numsamples = args.numsamples
num_epochs = args.num_epochs
px = args.patch_x
py = args.patch_y

#############
saving_path = f'./TwoDminGRU/L_{L}/dh_{dh}/numlayers_{num_layers}'
if not os.path.exists(saving_path):
    os.makedirs(saving_path+'/Outputs/')
    os.makedirs(saving_path+'/Params/')

savename = "_px"+str(px)+"_py"+str(py)+"_dh"+str(dh)+"_dm"+str(dmodel)+"_numsamples"+str(numsamples)+"_lr"+str(args.lr)+"_lrdecaytime"+str(args.lrdecaytime)+"_RNNsym"+str(RNNsymmetry)

########## Backup code ############
with open(__file__, 'r') as current_file:
    code_content = current_file.read()
with open(saving_path+'/train_backup'+savename+'.py', 'w') as backup_file:
    backup_file.write(code_content)

with open("./model.py", 'r') as current_file:
    code_content = current_file.read()
with open(saving_path+'/model_backup'+savename+'.py', 'w') as backup_file:
    backup_file.write(code_content)

with open("./Helper_functions.py", 'r') as current_file:
    code_content = current_file.read()
with open(saving_path+'/Helper_functions_backup'+savename+'.py', 'w') as backup_file:
    backup_file.write(code_content)

# Optional: if a launch script exists, keep a copy for reproducibility.
launch_script = "./run.sh"
if os.path.exists(launch_script):
    with open(launch_script, 'r') as current_file:
        code_content = current_file.read()
    with open(saving_path+'/run'+savename+'.sh', 'w') as backup_file:
        backup_file.write(code_content)
###############

model = TwoDFastGRU(d_hidden=dh, d_model=dmodel, n_layers=num_layers, patch_x = px, patch_y = py)

key1, key2 = jax.random.split(jax.random.key(1))
x = jax.random.randint(key1, (5000,L,L), 0, 2) # Dummy input data
# print(x)
params = model.init(key2, x) # Initialization call
param_count = sum(x.size for x in jax.tree_leaves(params))
print("Total number of parameters: ", param_count)
# JIT forward paths for fair timing.
parallel_apply = jit(lambda p, s: model.apply(p, s))
sequential_apply = jit(lambda p, s: model.apply(p, s, method="sequential_call"))

start = time.time()
y_parallel = parallel_apply(params, x)
y_parallel.block_until_ready()
print("parallel call ", time.time() - start)

start = time.time()
y_sequential = sequential_apply(params, x)
y_sequential.block_until_ready()
print("sequencial call ", time.time() - start)

start = time.time()
y_parallel = parallel_apply(params, x)
y_parallel.block_until_ready()
print(y_parallel)
print("parallel call ", time.time() - start)

start = time.time()
y_sequential = sequential_apply(params, x)
y_sequential.block_until_ready()
print(y_sequential)
print("sequencial call ", time.time() - start)

start = time.time()
y_parallel = parallel_apply(params, x)
y_parallel.block_until_ready()
print(y_parallel)
print("parallel call ", time.time() - start)

start = time.time()
y_sequential = sequential_apply(params, x)
y_sequential.block_until_ready()
print(y_sequential)
print("sequencial call ", time.time() - start)

# @jit
# @partial(jit, static_argnums=(2,3,4,8))
def get_loss(params, key, numsamples, Nx,Ny, model, RNNsymmetry):

    samples = model.apply(params,key,numsamples,Nx,Ny, method="sample")

    if RNNsymmetry == "nosym":
        log_probs = model.apply(params,samples)
    elif RNNsymmetry == "c4v":
        log_probs = model.apply(params,samples,method="logprobs_c4vsym")

    e_loc = jax.lax.stop_gradient(local_energy(samples, params, model, 0.5*log_probs, RNNsymmetry))
    e_avg = e_loc.mean()

    loss = jnp.mean(jnp.multiply(log_probs, e_loc) - jnp.multiply(e_avg, log_probs))
    return loss, e_loc


def inverse_schedule(step):
    return args.lr/(1+step/args.lrdecaytime)
optimizer = optax.adam(learning_rate=inverse_schedule)
opt_state = optimizer.init(params)
Nx = L
Ny = L
N = Nx*Ny
# queue_samples = jnp.zeros((2*Nx,Ny,numsamples, Nx,Ny), dtype = sim_dtype)
# offdiag_logpsi = jnp.zeros((2*N*numsamples), dtype = sim_dtype)
rng_key = jax.random.key(1)

@partial(jit, static_argnums=(4,5,6,7,8))
def step(params, rng_key, opt_state, model = model, numsamples = numsamples, Nx = Nx,Ny = Ny, RNNsymmetry = RNNsymmetry, get_loss=get_loss):
    rng_key, new_key = jax.random.split(rng_key)
    value, grads = jax.value_and_grad(get_loss, has_aux=True)(params, new_key, numsamples, Nx,Ny, model, RNNsymmetry)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, value, new_key

####### Checkpoint #######

try:
    # Attempt to load the checkpoint
    with open(saving_path+'/Params/checkpoint'+savename+'.pkl', 'rb') as f:
        checkpoint_data = pickle.load(f)
    print(checkpoint_data.keys())
    # Extract each variable from the checkpoint
    params = checkpoint_data["model_state"]
    opt_state = checkpoint_data["optimizer_state"]
    rng_key = checkpoint_data["rng"]
    initial_epoch = checkpoint_data["epoch"]
    energies = checkpoint_data["energies"]
    variances = checkpoint_data["variances"]
    durations = checkpoint_data["durations"]
    
    print("Checkpoint loaded successfully.")

except FileNotFoundError:
    # Initialize variables if the checkpoint does not exist
    print("Checkpoint not found. Initializing variables.")
    
    initial_epoch = 0                          # Start from epoch 0
    energies, variances, durations = [], [], []

###################

if args.dotraining:
    print('Training started')
    for epoch in range(initial_epoch,num_epochs):

        ########### Checkpoint before optimizing for reproducibility ###########
        if epoch % 1000 == 0:
            checkpoint_data = {
                "model_state": params,  # Model state dictionary
                "optimizer_state": opt_state,  # Optimizer state
                "rng": rng_key,
                "epoch": epoch,  # Current epoch
                "energies": energies,
                "variances": variances,
                "durations": durations
            }
            with open(saving_path+'/Params/checkpoint'+savename+'.pkl', 'wb') as f:
                pickle.dump(checkpoint_data, f)
        ################

        s = time.perf_counter()
        params, opt_state, (_, eloc), rng_key = step(params, rng_key, opt_state)
        # Force device work to finish before timing ends
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, (params, opt_state, eloc))

        e = time.perf_counter()
        duration = e - s
        durations.append(duration)

        energies.append(jnp.mean(eloc))
        variances.append(jnp.var(eloc))


        if epoch % 10 == 0:
            print("Step = ",epoch, ", Energy =", jnp.mean(eloc), ", Var =", jnp.var(eloc))
            print("Duration =", durations[-1], "\n")

            np.savetxt(saving_path+"/energies"+savename+".txt", energies)
            np.savetxt(saving_path+"/variances"+savename+".txt", variances)
            # ######
            # gpu_memory = get_gpu_memory()
            # for idx, gpu in enumerate(gpu_memory):
            #     print(f"GPU {idx}:")
            #     print(f"  Total Memory: {gpu['total_memory_gb']:.2f} GB")
            #     print(f"  Used Memory: {gpu['used_memory_gb']:.2f} GB")
            #     print(f"  Free Memory: {gpu['free_memory_gb']:.2f} GB")
            # ########

    dict_0 = {'Energies': [float(x) for x in energies],'Variance': [float(x) for x in variances], 'Time': durations}
    df = pd.DataFrame(dict_0)
    df.to_csv(saving_path+'/Outputs/outputs'+savename+'.csv')

    checkpoint_data = {
        "model_state": params,  # Model state dictionary
        "optimizer_state": opt_state,  # Optimizer state
        "rng": rng_key,
        "epoch": epoch,  # Current epoch
        "energies": energies,
        "variances": variances,
        "durations": durations
    }

    with open(saving_path+'/Params/checkpoint'+savename+'.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)

def final_energy(params, key, model, Nx, Ny, num_samples_final, max_batch_size = numsamples):

    N = Nx*Ny
    batch_size = max_batch_size

    batch_steps = ceil(num_samples_final//batch_size)
    samples = jnp.zeros((batch_size, Nx, Ny))
    e_loc = jnp.zeros((num_samples_final))
    log_probs = jnp.zeros((batch_size))

    keys = jax.random.split(key, batch_steps)
 
    for i in range(batch_steps):
        print(i+1, "/", batch_steps)
        samples = model.apply(params,keys[i], batch_size, Nx,Ny, method="sample")
        if RNNsymmetry == "nosym":
            log_probs = model.apply(params,samples)
        elif RNNsymmetry == "c4v":
            log_probs = model.apply(params,samples,method="logprobs_c4vsym")

        e_loc = e_loc.at[i*batch_size:(i+1)*batch_size].set(local_energy(samples, params, model, 0.5*log_probs, RNNsymmetry))
    
    return jnp.mean(e_loc), jnp.var(e_loc), jnp.sqrt(jnp.var(e_loc))/jnp.sqrt(num_samples_final), num_samples_final

duration = time.time()-starttime
print(f'Job took: {duration} seconds')

# num_samples_final = 1000000
num_samples_final = 100000
final_data = final_energy(params, rng_key, model, Nx, Ny, num_samples_final, max_batch_size=1000)

print(final_data)
dict_1 = {'mean energy': [final_data[0]],'Variance': [final_data[1]], 'Error': [final_data[2]], 'Numsamples': [final_data[3]], 'duration': [duration]}
df2 = pd.DataFrame(dict_1)
df2.to_csv(saving_path+'/Outputs/finaldata'+savename+'.csv')
