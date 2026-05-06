import jax
import jax.numpy as jnp
from functools import partial
sim_dtype = jnp.float32
DEFAULT_MAX_PARALLEL_FLIPPED_STATES = 4096
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
# @partial(jax.jit, static_argnums=(6,))
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


# def local_energy(
#     samples,
#     params,
#     model,
#     log_psi,
#     RNNsymmetry,
#     max_parallel_flipped_states=DEFAULT_MAX_PARALLEL_FLIPPED_STATES,
# ) -> List[float]:
#     """Computes the local energy of the 2D Heisenberg model"""
#     numsamples,Nx,Ny = samples.shape
#     N = Nx*Ny
#     # print("Hello")

#     local_energies = jnp.zeros((numsamples), dtype = sim_dtype)
#     max_parallel_flipped_states = max(int(max_parallel_flipped_states), int(numsamples))
#     bonds_per_chunk = max(1, max_parallel_flipped_states // int(numsamples))

#     for i in range(Nx-1): #diagonal elements (right neighbours)
#         spins_products = 0.25*(2*samples[:,i]-1)*(2*samples[:,i+1]-1)
#         local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

#     for j in range(Ny-1): #diagonal elements (upward neighbours (or downward, it depends on the way you see the lattice))
#         spins_products = 0.25*(2*samples[:,:,j]-1)*(2*samples[:,:,j+1]-1)
#         local_energies += jnp.sum(jnp.copy(spins_products), axis = 1)

#     def batched_flipped_logpsi(flipped_states):
#         if RNNsymmetry == "nosym":
#             return 0.5*model.apply(params,flipped_states)
#         elif RNNsymmetry == "c4v":
#             return 0.5*model.apply(params,flipped_states,method = "logprobs_c4vsym")
#         raise ValueError(f"Unknown RNNsymmetry={RNNsymmetry}")

#     def flip_one_bond(i1, j1, i2, j2):
#         flipped_state = samples.at[:, i1,j1].set(1 - samples[:, i1,j1])
#         flipped_state = flipped_state.at[:, i2,j2].set(1 - flipped_state[:, i2,j2])
#         return flipped_state

#     def accumulate_off_diag(bonds, output):
#         for start in range(0, len(bonds), bonds_per_chunk):
#             bond_chunk = bonds[start : start + bonds_per_chunk]
#             i1 = jnp.asarray([bond[0][0] for bond in bond_chunk], dtype=jnp.int32)
#             j1 = jnp.asarray([bond[0][1] for bond in bond_chunk], dtype=jnp.int32)
#             i2 = jnp.asarray([bond[1][0] for bond in bond_chunk], dtype=jnp.int32)
#             j2 = jnp.asarray([bond[1][1] for bond in bond_chunk], dtype=jnp.int32)

#             spins_i = samples[:, i1, j1].T
#             spins_j = samples[:, i2, j2].T
#             flippable = (spins_i + spins_j) == 1

#             flipped_states = jax.vmap(flip_one_bond)(i1, j1, i2, j2).reshape(-1, Nx, Ny)
#             flipped_logpsi = batched_flipped_logpsi(flipped_states).reshape(len(bond_chunk), numsamples)
#             output += jnp.sum(flippable * (-0.5) * jnp.exp(flipped_logpsi - log_psi[None, :]), axis=0)
#         return output

#     horizontal_bonds = [((i, j), (i + 1, j)) for i in range(Nx - 1) for j in range(Ny)]
#     vertical_bonds = [((i, j), (i, j + 1)) for j in range(Ny - 1) for i in range(Nx)]

#     off_diag_term_horizontal = accumulate_off_diag(horizontal_bonds, jnp.zeros((numsamples), dtype=sim_dtype))
#     off_diag_term_vertical = accumulate_off_diag(vertical_bonds, jnp.zeros((numsamples), dtype=sim_dtype))

#     # print("Hello again")

#     local_energies += off_diag_term_vertical +  off_diag_term_horizontal
#     return local_energies


#########


def log_probs_fun(params, model, samples):
    return 0.5*model.apply(params,samples)

def get_minSR_gradients(params, model, samples, local_energies):
  jacobian = jax.jacrev(log_probs_fun)(params, model, samples)

  numsamples = samples.shape[0]

  flattened_jac, tree = jax.tree_util.tree_flatten(jacobian)

  shapes = [it.shape for it in flattened_jac]

  slices = []
  last = flattened_jac[0][0].size
  slices.append(slice(0,last))
  for it in flattened_jac[1:]:
      slices.append(slice(last,last+it[0].size))
      last += it[0].size

  jac = jnp.concatenate([it.reshape(it.shape[0],-1) for it in flattened_jac], axis=-1)
  jac -= jnp.mean(jac, axis = 0)
  jac = jac/ jnp.sqrt(numsamples)
  XdaggerX = jac @ jac.T

  XdaggerX_inv = jax.scipy.linalg.inv((XdaggerX + 1e-2 * jnp.eye(XdaggerX.shape[0]))) ### Choosing 1e-2 helps to stabilize the training

  gradients = jac.T @ XdaggerX_inv @ local_energies * ( 2 / jnp.sqrt(numsamples))

  ### unflatten
  flat_tree = []
  for shape, _slice in zip(shapes, slices):
      flat_tree.append(gradients[_slice].reshape(shape[1:]))

  original_grad = jax.tree_util.tree_unflatten(tree, flat_tree)

#   return original_grad
  return original_grad, XdaggerX


def get_grad(params, key, numsamples, Nx,Ny, model, RNNsymmetry):
    samples = model.apply(params,key,numsamples,Nx,Ny,method="sample") # This line with the next one take ~18.62it/s for N = 20 1DTFIM
    log_probs = model.apply(params,samples)
    e_loc = local_energy(samples, params, model, 0.5*log_probs,RNNsymmetry)
    e_loc_c = e_loc - e_loc.mean()
    grads, XdaggerX = get_minSR_gradients(params, model, samples, e_loc_c)
    return grads, e_loc, XdaggerX
