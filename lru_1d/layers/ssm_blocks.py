from typing import Any, TypeVar

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.typing import Dtype

from .ssm_1d import SSM1D

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any



class AutoregressiveSSMBlock(Module):
    def __init__(
        self,
        ssm: SSM1D,
        *,
        skip: bool = False,
        rngs: rnglib.Rngs,
        param_dtype: Dtype = jnp.float32,
        linear_layer: nnx.Linear | None = None,
    ):
        self.skip = skip
        self.param_dtype = param_dtype
        if isinstance(ssm, SSM1D):
            dim = ssm.cell.in_features
        else:
            dim = ssm.ssm_x.cell.in_features
            
        self.ssm = ssm
        if linear_layer is None:
            self.linear = nnx.Linear(dim, 2*dim, rngs=rngs, param_dtype=self.param_dtype)
        else:
            assert linear_layer.in_features == dim, "wrong number of in_features!"
            assert linear_layer.out_features == 2*dim, "wrong number of out_features!"
            self.linear = linear_layer


    def __call__(self, inputs: Array,
        *,
        parallel: bool | None = None,
    ):
        carry, ssm_output = self.ssm(inputs, parallel=parallel)
        output = nnx.glu(self.linear(ssm_output))
        if self.skip:
            output += inputs

        return carry, output
    
    def iter(self, 
        input_carries: tuple[Carry, ...],
        inputs: tuple[Array, ...],
        skip_input: Array | None = None
    ) -> tuple[Carry, Array]:
        carry, ssm_output = self.ssm.iter(input_carries, inputs) # type: ignore
        output = nnx.glu(self.linear(ssm_output))
        if self.skip and skip_input is not None:
            # Since the inputs are from different sites than output,
            # we can't directly add them as skip connections, 
            # so we need to rely on the caller to pass in the appropriate 
            # skip connection (i.e. the one from the same site as output).
            # This is a bit inelegant but it allows us to keep the skip connection
            # without having to worry about the snake ordering in the block itself.
            output += skip_input
        return carry, output
    

def _tuple_index(xs: tuple[Array, ...], i: int, axis: int=0) -> tuple[Array, ...]:
    return tuple(
        jax.lax.index_in_dim(operand, i, axis=axis, keepdims=False)
        for operand in xs
    )


class AutoregressiveBlockSequence(Module):
    def __init__(
        self,
        layers: list[AutoregressiveSSMBlock]
    ):
        self.layers = nnx.List(layers)

    def __call__(self, inputs: Array,
        *,
        parallel: bool | None = None,
    ):
        outputs = inputs
        for layer in self.layers:
            _, outputs = layer(outputs, parallel=parallel)
        return outputs
    
    def iter(self,
             input_carries: tuple[Carry, ...],
             inputs: tuple[Array, ...]
    ) -> tuple[Carry, Array]:
        carries = jnp.zeros_like(input_carries[0])
        outputs = jnp.zeros_like(inputs[0])
        for l, layer in enumerate(self.layers):
            carry_l, output_l = layer.iter(_tuple_index(input_carries, l, axis=0),
                                           _tuple_index(inputs, l, axis=0),
                                           skip_input=outputs[l-1, ...] if l > 0 else None)
            carries = carries.at[l, ...].set(carry_l)
            outputs = outputs.at[l, ...].set(output_l)
            
        return carries, outputs
    