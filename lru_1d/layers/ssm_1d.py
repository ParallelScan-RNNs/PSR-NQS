from typing import Any, TypeVar
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx.module import Module
from flax.nnx.transforms import iteration

from .lru_cell import LRUCell

A = TypeVar("A")
Array = jax.Array
Output = Any
Carry = Any


def flip_inputs(time_axis):
    return jax.jit(lambda x: jnp.flip(x, axis=time_axis))

@jax.jit
def identity(x: A) -> A:
    return x



class SSM1D(Module):
    """The ``SSM1D`` module takes any :class:`LRUCell` instance and applies it over a sequence
    using :func:`flax.nnx.scan` or :func:`jax.lax.associative_scan`.
    """

    state_axes: dict[str, int | type[iteration.Carry] | None]

    def __init__(
        self,
        cell: LRUCell,
        *,
        time_axis: int = 0,
        reverse: bool = False,
        keep_order: bool = True,
        parallel: bool = True,
        unroll: int | bool = 1,
        state_axes: Mapping[str, int | type[iteration.Carry] | None] | None = None
    ):
        self.cell = cell
        self.time_axis = time_axis
        self.reverse = reverse
        self.keep_order = keep_order
        self.parallel = parallel
        self.unroll = unroll
        self.state_axes = state_axes or nnx.StateAxes({...: iteration.Carry})  # type: ignore

    def binary_operator_diag(self, element_i, element_j):
        # Binary operator for parallel scan of linear recurrence.
        a_i, bu_i = element_i
        a_j, bu_j = element_j
        return a_j * a_i, a_j * bu_i + bu_j
    
    def binary_operator_diag_complex(self, element_i, element_j):
        a_i_re, a_i_im, bu_i_re, bu_i_im = element_i
        a_j_re, a_j_im, bu_j_re, bu_j_im = element_j
        
        # a_j * a_i
        a_re = a_j_re * a_i_re - a_j_im * a_i_im
        a_im = a_j_re * a_i_im + a_j_im * a_i_re
        
        # a_j * bu_i + bu_j
        bu_re = a_j_re * bu_i_re - a_j_im * bu_i_im + bu_j_re
        bu_im = a_j_re * bu_i_im + a_j_im * bu_i_re + bu_j_im
        
        return a_re, a_im, bu_re, bu_im

    def iter(self, carries: tuple[Carry], inputs: tuple[Array]) -> tuple[Carry, Array]:
        """Single step of the SSM over one time step."""
        return self.cell(carries[0], inputs[0])
    
    def initialize_carry(self, input_shape: tuple[int, ...], drop_axis: int | None = None) -> Carry:
        if drop_axis is not None:
            input_shape = input_shape[:drop_axis] + input_shape[drop_axis + 1 :]
        return self.cell.initialize_carry(input_shape)
    
    def initialize_carries(self, input_shape: tuple[int, ...], drop_axis: int | None = None) -> tuple[Carry]:
        return (self.initialize_carry(input_shape, drop_axis=drop_axis),)

    def __call__(
        self,
        inputs: Array,                        # [Lx, *batch..., input_features]
        initial_carry: Carry | None = None,   # [*batch..., hidden_features]
        partial_carries: Array | None = None, # [Lx, *batch..., hidden_features]
        time_axis: int | None = None,
        reverse: bool | None = None,
        keep_order: bool | None = None,
        parallel: bool | None = None,
        unroll: int | bool | None = None,
        shift_input: bool = True,
    ):
        time_axis = time_axis if time_axis is not None else self.time_axis
        reverse = reverse if reverse is not None else self.reverse
        parallel = parallel if parallel is not None else self.parallel
        keep_order = keep_order if keep_order is not None else self.keep_order
        
        # maybe reverse the sequence
        inputs = jax.lax.cond(reverse, flip_inputs(time_axis), identity, inputs)
        if partial_carries is not None:
            partial_carries = jax.lax.cond(reverse, flip_inputs(time_axis), identity, partial_carries)
            
        if shift_input:
            x0 = jnp.zeros_like(jax.lax.index_in_dim(inputs, 0, axis=time_axis, keepdims=True))
            inputs = jnp.concatenate(
                [x0, jax.lax.slice_in_dim(inputs, 0, inputs.shape[time_axis]-1, axis=time_axis)], 
                axis=time_axis)

        carry: Carry = (self.initialize_carry(inputs.shape, drop_axis=time_axis)
                        if initial_carry is None 
                        else initial_carry)

        if parallel:
            carries, outputs = self._parallel_scan(carry, inputs, partial_carries, time_axis, unroll)
        else:
            carries, outputs = self._sequential_scan(carry, inputs, partial_carries, time_axis, unroll)

        if keep_order:
            carries = jax.lax.cond(reverse, flip_inputs(time_axis), identity, carries)
            outputs = jax.lax.cond(reverse, flip_inputs(time_axis), identity, outputs)

        return carries, outputs
    
    
    def _sequential_scan(self, carry: Carry, inputs: Array,
                         partial_carries: Array | None, 
                         time_axis: int, 
                         unroll: int | bool | None) -> tuple[Carry, Array]:
        state_axes = iteration.StateAxes({**self.state_axes})  # type: ignore[misc]
        unroll = unroll if unroll is not None else self.unroll
        pc_axis = time_axis if partial_carries is not None else None
    
        @nnx.scan(
            in_axes=(state_axes, iteration.Carry, time_axis, pc_axis),
            out_axes=(iteration.Carry, (time_axis, time_axis)),
            unroll=unroll,
        )
        def scan_fn(
            cell: LRUCell, carry: Carry, input: Array, partial_carry: Array | None,
        ) -> tuple[Carry, tuple[Carry, Array]]:
            carry, output = cell(carry, input, partial_carry)
            return carry, (carry, output)

        _, (carries, outputs) = scan_fn(self.cell, carry, inputs, partial_carries)
        return carries, outputs

    def _parallel_scan(self, carry: Carry, inputs: Array,
                       partial_carries: Array | None, 
                       time_axis: int,
                       unroll: int | bool | None) -> tuple[Carry, Array]:
        
        Lambda = jnp.expand_dims(self.cell.get_recurrent_kernel(), 
                                 axis=range(jnp.ndim(inputs) - 1))
            
        Lambda_elements = jnp.repeat(Lambda, 1+inputs.shape[time_axis], axis=time_axis)
        init_carry = jnp.expand_dims(carry, axis=time_axis)
        Bu_elements = jnp.concatenate([init_carry, self.cell.transform_inputs(inputs, partial_carries)], 
                                      axis=time_axis)
        _, _, carries_re, carries_im = jax.lax.associative_scan(self.binary_operator_diag_complex, 
                                              (Lambda_elements.real, Lambda_elements.imag, Bu_elements.real, Bu_elements.imag), 
                                              axis=time_axis)
        
        carries_re = jax.lax.slice_in_dim(carries_re, 1, carries_re.shape[time_axis], axis=time_axis)
        carries_im = jax.lax.slice_in_dim(carries_im, 1, carries_im.shape[time_axis], axis=time_axis)
        carries = jax.lax.complex(carries_re, carries_im)
        outputs = self.cell.compute_output(carries, inputs, partial_carries)

        return carries, outputs
