from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import rnglib
from flax.typing import Dtype

from ..layers import AutoregressiveSSMBlock, SSM1D, AutoregressiveBlockSequence, LRUCell

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any


class PositiveWavefunctionOutput(nnx.Module):
    def __init__(self, emb_dim: int, local_dim: int, 
                 rngs: rnglib.Rngs,
                 param_dtype: Dtype = jnp.float32):
        self.magnitude = nnx.Linear(emb_dim, local_dim, param_dtype=param_dtype, rngs=rngs)

    def __call__(self, y):
        # y is of shape (..., emb_dim)
        # we want to return an array of shape (..., local_dim)
        y = self.logits(y)
        return nnx.log_softmax(y, axis=-1)/2

    def logits(self, ssm_output):
        return self.magnitude(ssm_output)
    
    def phase(self, ssm_output):
        return jnp.zeros_like(ssm_output, 
                              shape=(*ssm_output.shape[:-1], 
                                     self.magnitude.out_features))
        
    def log_psi_to_log_prob(self, log_psi):
        return 2*jnp.real(log_psi)
    
    def log_psi(self, ssm_output):
        return self(ssm_output)
    
    def log_prob(self, ssm_output):
        log_psi_ = self(ssm_output)
        return self.log_psi_to_log_prob(log_psi_)
    
    def sample(self, ssm_output, rngs: rnglib.RngStream):
        log_psi_ = self(ssm_output)
        log_probs = self.log_psi_to_log_prob(log_psi_)
        sample = rngs.categorical(log_probs, axis=-1)
        return sample, log_psi_
    
    @property
    def output_dtype(self):
        return self.magnitude.param_dtype



class Positive1DSSMWavefunction(nnx.Module):
    def __init__(
        self,
        local_dim: int,
        emb_dim: int,
        ssm_dim: int,
        num_layers: int = 1,
        *,
        param_dtype: Dtype = jnp.float32,
        parallel: bool = True,
        unroll: int | bool = 1,
        rngs: rnglib.Rngs | None = None,
    ):
        self.local_dim = local_dim
        self.emb_dim = emb_dim
        self.ssm_dim = ssm_dim
        self.num_layers = num_layers
        self.parallel = parallel
        self.param_dtype = param_dtype
        self.rngs: rnglib.Rngs
        if rngs is None:
            self.rngs = rnglib.Rngs(0, params=1234, sampling=5678)
        elif isinstance(rngs, rnglib.Rngs):
            self.rngs = rngs
        else:
            raise ValueError(
                'Expected rngs to be a jax.Array, int, Rngs, or bool. '
                f'Got {type(rngs)}.'
            )
        self.emb = nnx.Embed(local_dim, emb_dim, param_dtype=self.param_dtype, rngs=self.rngs)
        self.layers = AutoregressiveBlockSequence(
            [AutoregressiveSSMBlock(
                SSM1D(LRUCell(self.emb_dim, self.ssm_dim, param_dtype=self.param_dtype, rngs=self.rngs), parallel=self.parallel, unroll=unroll),
                skip=(i > 0), rngs=self.rngs, param_dtype=self.param_dtype) 
             for i in range(num_layers)])
        self.output_layer = PositiveWavefunctionOutput(emb_dim, local_dim, param_dtype=self.param_dtype, rngs=self.rngs)


    def __call__(self, inputs: Array, parallel: bool | None = None):
        if parallel is None:
            parallel = self.parallel
            
        embedded = self.emb(inputs)
        ssm_outputs = self.layers(embedded, parallel=parallel)
        cond_log_psis = self.output_layer.log_psi(ssm_outputs)
        return cond_log_psis
    
    @staticmethod
    def generate_path(lattice_shape, reverse=False):
        Lx = lattice_shape[0]
        r = int(reverse)
        return [((i,), (i-(-1)**r,)) for i in range(Lx)[::(-1)**r]]

    def last_index(self, lattice_shape, reverse=False):
        return (0,) if reverse else (lattice_shape[0]-1,)
    
    def sample(self, lattice_shape, nsamples,
        rngs: rnglib.Rngs | rnglib.RngStream | None = None,
    ):
        if rngs is None:
            rngs = self.rngs
        if isinstance(rngs, rnglib.Rngs):
            rngs = rngs.sampling

        Lx = lattice_shape[0]
        shape = (Lx, nsamples)
        init_carry = self.layers.layers[0].ssm.initialize_carry((nsamples, self.ssm_dim))
        init_input = jnp.zeros((nsamples, self.emb_dim), dtype=self.param_dtype)

        inputs = jnp.zeros((self.num_layers+1, *shape, self.emb_dim), dtype=self.param_dtype)
        carries = jnp.stack([l.ssm.initialize_carry((*shape, self.ssm_dim)) for l in self.layers.layers], axis=0)

        samples = jnp.zeros(shape, dtype=int)
        cond_log_psis = jnp.zeros((*shape, self.local_dim), dtype=self.output_layer.output_dtype)
        for ((i,), (i_prev,)) in self.generate_path(lattice_shape):
            for l, layer in enumerate(self.layers.layers):
                carry_x = carries[l, i_prev, ...] if 0 <= i_prev < Lx else init_carry
                input_x = inputs[l, i_prev, ...] if 0 <= i_prev < Lx else init_input

                carry_l, output_l = layer.iter((carry_x,), (input_x,), skip_input=inputs[l, i, ...])
                carries = carries.at[l, i, ...].set(carry_l)
                inputs = inputs.at[l+1, i, ...].set(output_l)

            log_psi = self.output_layer.log_psi(inputs[-1, i, ...])
            cond_log_psis = cond_log_psis.at[i, ...].set(log_psi)

            log_probs = self.output_layer.log_psi_to_log_prob(log_psi)
            s = rngs.categorical(log_probs, axis=-1)
            samples = samples.at[i, ...].set(s)
            inputs = inputs.at[0, i, ...].set(self.emb(s))

        return samples, cond_log_psis

    @property
    def num_spatial_dims(self):
        return 1

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.num_spatial_dims))
    
    def spatial_shape(self, inputs: Array) -> tuple[int, ...]:
        return inputs.shape[:self.num_spatial_dims]

    def select_lp(self, inputs: Array, clp: Array) -> Array:
        return jnp.take_along_axis(clp, inputs[..., None], axis=-1).squeeze(-1)

    def conditional_log_psis(self, inputs: Array, parallel: bool | None = None) -> Array:
        return self(inputs, parallel=parallel)
    
    def conditional_log_probs(self, inputs: Array, parallel: bool | None = None) -> Array:
        log_psis = self(inputs, parallel=parallel)
        return self.output_layer.log_psi_to_log_prob(log_psis)

    def log_psis_by_site(self, inputs, cond_log_psis=None, parallel: bool | None = None):
        cond_log_psis = cond_log_psis if cond_log_psis is not None else self.conditional_log_psis(inputs, parallel=parallel)
        return self.select_lp(inputs, cond_log_psis)
    
    def log_probs_by_site(self, inputs, cond_log_psis=None, parallel: bool | None = None):
        log_psis = self.log_psis_by_site(inputs, cond_log_psis=cond_log_psis, parallel=parallel)
        return self.output_layer.log_psi_to_log_prob(log_psis)
    
    def log_psis(self, inputs, cond_log_psis=None, parallel: bool | None = None):
        return self.log_psis_by_site(inputs, cond_log_psis=cond_log_psis, parallel=parallel).sum(self.spatial_axes)
    
    def log_probs(self, inputs, cond_log_psis=None, parallel: bool | None = None):
        log_psis = self.log_psis(inputs, cond_log_psis=cond_log_psis, parallel=parallel)
        return self.output_layer.log_psi_to_log_prob(log_psis)

