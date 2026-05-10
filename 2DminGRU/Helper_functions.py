import jax
import jax.numpy as jnp
from functools import partial
sim_dtype = jnp.float32
from typing import List, Tuple, Union, Optional, Callable, Any
from math import ceil
import subprocess

#######################################################################
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
######################################################################

###########
def get_gpu_memory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    # Parse output
    memory_info = result.stdout.strip().split('\n')
    memory_data = []
    for line in memory_info:
        total, used, free = map(int, line.split(', '))
        memory_data.append({
            'total_memory_gb': total / 1024,
            'used_memory_gb': used / 1024,
            'free_memory_gb': free / 1024
        })
    return memory_data
###########


############
def local_energy(samples, params, model, log_psi, RNNsymmetry, forward_mode="parallel") -> List[float]:
    """Original bond-by-bond local energy implementation."""
    numsamples,Nx,Ny = samples.shape

    local_energies = jnp.zeros((numsamples), dtype = sim_dtype)

    for i in range(Nx-1): #diagonal elements (right neighbours)
        spins_products = 0.25*(2*samples[:,i]-1)*(2*samples[:,i+1]-1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    for j in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice))
        spins_products = 0.25*(2*samples[:,:,j]-1)*(2*samples[:,:,j+1]-1)
        local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

    def evaluate_logpsi(flipped_state):
        if RNNsymmetry == "nosym":
            if forward_mode == "parallel":
                return 0.5 * model.apply(params, flipped_state)
            if forward_mode == "sequential":
                return 0.5 * model.apply(params, flipped_state, method="sequential_call")
            raise ValueError(f"Unknown forward_mode={forward_mode}")
        if RNNsymmetry == "c4v":
            return 0.5 * model.apply(params, flipped_state, forward_mode, method="logprobs_c4vsym")
        raise ValueError(f"Unknown RNNsymmetry={RNNsymmetry}")

    def step_fn_horizontal(n, state):
        s, output = state
        _, Nx,Ny = s.shape

        i = (n//Ny) #set back to zero when equal to Nx-1
        j = n%Ny

        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i+1,j].set(1 - flipped_state[:, i+1,j])
        flipped_logpsi = evaluate_logpsi(flipped_state)

        output += (s[:, i,j] + s[:, i+1,j] == 1) *(-0.5)* jnp.exp(flipped_logpsi - log_psi)

        return s, output

    def step_fn_vertical(n, state):
        s, output = state
        _, Nx,Ny = s.shape

        j = (n//Nx) #set back to zero when equal to Nx-1
        i = n%Nx

        flipped_state = s.at[:, i,j].set(1 - s[:, i,j])
        flipped_state = flipped_state.at[:, i,j+1].set(1 - flipped_state[:, i,j+1])
        flipped_logpsi = evaluate_logpsi(flipped_state)

        output += ((s[:, i,j] + s[:, i,j+1] == 1)*(-0.5))*jnp.exp(flipped_logpsi - log_psi)

        return s, output

    output = jnp.zeros((numsamples), dtype=sim_dtype)
    _, off_diag_term_vertical = jax.lax.fori_loop(0, Nx*(Ny-1), step_fn_vertical, (samples, output))
    _, off_diag_term_horizontal = jax.lax.fori_loop(0, (Nx-1)*(Ny), step_fn_horizontal, (samples, output))

    local_energies += off_diag_term_vertical +  off_diag_term_horizontal
    return local_energies

#########
