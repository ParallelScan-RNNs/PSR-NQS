from typing import Any

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import Dtype, Initializer, PromoteDtypeFn, Shape


Array = jax.Array
Output = Any
Carry = Any



def nu_init(r_min: float, r_max: float):
    def fn(rng: Array, shape: Shape, dtype: Dtype):
        u1 = jax.random.uniform(rng, shape=shape, dtype=dtype)
        nu = jnp.log(u1*(r_max**2 - r_min**2) + r_min**2)
        nu_log = jnp.log(-0.5*nu)
        return nu_log
    return fn

def theta_init(max_phase: float):
    def fn(rng: Array, shape: Shape, dtype: Dtype):
        u2 = jax.random.uniform(rng, shape=shape, dtype=dtype)
        theta_log = jnp.log(max_phase * u2)
        return theta_log
    return fn
    
def gamma_init(nu_log: Array, theta_log: Array):
    # Normalization factor
    diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
    gamma_log = jnp.log(jnp.abs(1 - jnp.abs(diag_lambda)**2))/2
    return gamma_log


cplx_type_map = {jnp.float32: jnp.complex64, jnp.float64: jnp.complex128}


class LRUDenseOutput(nnx.Module):
    
    def __init__(self, 
        in_features: int, 
        hidden_features: int,
        *,
        c_initializer: Initializer,
        d_initializer: Initializer,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        keep_rngs: bool = False,
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = promote_dtype
        self.rngs: rnglib.RngStream | None
        if keep_rngs:
            self.rngs = rngs.carry.fork()
        else:
            self.rngs = nnx.data(None)

        self.C_re = nnx.Linear(in_features=self.hidden_features, 
                               out_features=self.in_features, 
                               use_bias=False, 
                               kernel_init=c_initializer,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype, 
                               promote_dtype=self.promote_dtype,
                               rngs=rngs)
        self.C_im = nnx.Linear(in_features=self.hidden_features, 
                               out_features=self.in_features,
                               use_bias=False, 
                               kernel_init=c_initializer,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype, 
                               promote_dtype=self.promote_dtype,
                               rngs=rngs)
        self.D = nnx.Param(d_initializer(rngs.params(), shape=(in_features,), dtype=self.param_dtype))
        
    def __call__(self, hidden_sequence: Array, input_sequence: Array) -> Array:
        h = self.C_re(hidden_sequence.real) - self.C_im(hidden_sequence.imag)
        return h + self.D * input_sequence




class LRUCell(nnx.Module):
    r"""LRU cell.

    The mathematical definition of the cell is as follows

    .. math::

        \begin{array}{ll}
        h' = A * h + gamma * B * x \\
        \end{array}

    where x is the input and h is the output of the previous time step.

    Args:
        in_features: number of input features.
        hidden_features: number of output features.
        nu_initializer: .
        theta_initializer: .
        dtype: the dtype of the computation (default: None).
        param_dtype: the dtype passed to parameter initializers (default: float32).
        keep_rngs: whether to store the input rngs as attribute (i.e. `self.rngs = rngs`)
          (default: True). If rngs is stored, we should split the module as
          `graphdef, params, nondiff = nnx.split(module, nnx.Param, ...)` where `nondiff`
          contains RNG object associated with stored `self.rngs`.
        rngs: rng key.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        nu_initializer: Initializer = nu_init(0.0, 1.0),
        theta_initializer: Initializer = theta_init(2*jnp.pi),
        b_initializer: Initializer = nnx.initializers.variance_scaling(0.5, "fan_out", "normal"),
        c_initializer: Initializer = nnx.initializers.variance_scaling(1.0, "fan_out", "normal"),
        d_initializer: Initializer = nnx.initializers.normal(stddev=1.0),
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.promote_dtype = promote_dtype

        self.nu_log = nnx.Param(nu_initializer(rngs.params(), shape=(hidden_features,), dtype=param_dtype))
        self.theta_log = nnx.Param(theta_initializer(rngs.params(), shape=(hidden_features,), dtype=param_dtype))
        self.gamma_log = nnx.Param(gamma_init(self.nu_log.value, self.theta_log.value))

        # Materializing the diagonal of Lambda and projections
        self.lambda_ = jnp.exp(-jnp.exp(self.nu_log) + 1j*jnp.exp(self.theta_log)) # type: ignore
        self.gamma = jnp.exp(self.gamma_log) # type: ignore

        self.B_re = nnx.Linear(in_features=self.in_features, 
                               out_features=self.hidden_features, 
                               use_bias=False, 
                               kernel_init=b_initializer,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype, 
                               promote_dtype=self.promote_dtype,
                               rngs=rngs)
        self.B_im = nnx.Linear(in_features=self.in_features, 
                               out_features=self.hidden_features, 
                               use_bias=False, 
                               kernel_init=b_initializer,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype, 
                               promote_dtype=self.promote_dtype,
                               rngs=rngs)

        self.output = LRUDenseOutput(in_features=self.in_features,
                                     hidden_features=self.hidden_features,
                                     c_initializer=c_initializer,
                                     d_initializer=d_initializer,
                                     dtype=self.dtype,
                                     param_dtype=self.param_dtype, 
                                     promote_dtype=self.promote_dtype,
                                     rngs=rngs)

    def __call__(self, carry: Array, inputs: Array, partial_carry: Array | None = None) -> tuple[Array, Array]:  # type: ignore[override]
        """Linear recurrent unit (LRU) cell.

        Args:
            carry: the hidden state of the LRU cell,
            initialized using ``LRUCell.initialize_carry``.
            inputs: an ndarray with the input for the current time step.
            partial_carry: an ndarray with the same dimension as the carry that will be added directly to the new carry
            All dimensions except the final are considered batch dimensions.

        Returns:
            A tuple with the new carry and the output.
        """
        new_carry = self.transform_carry(carry) + self.transform_inputs(inputs, partial_carry)
        return new_carry, self.compute_output(new_carry, inputs, partial_carry)

    def transform_inputs(self, inputs: Array, partial_carry: Array | None = None):
        B_outputs = self.B_re(inputs) + 1j*self.B_im(inputs)
        transformed_inputs = self.gamma * B_outputs
        if partial_carry is not None:
            return transformed_inputs + partial_carry
        else:
            return transformed_inputs

    def transform_carry(self, h: Array):
        return self.lambda_ * h
    
    def compute_output(self, carry: Array, inputs: Array, partial_carry: Array | None = None):
        if partial_carry is not None:
            return self.output(carry - partial_carry, inputs)
        else:
            return self.output(carry, inputs)
    
    def get_recurrent_kernel(self):
        return self.lambda_

    def initialize_carry(
        self,
        input_shape: tuple[int, ...]
    ) -> Array:  # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
            rngs: random number generator passed to the init_fn.
            input_shape: a tuple providing the shape of the input to the cell.

        Returns:
            An initialized carry for the given RNN cell.
        """
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)

        h = jnp.zeros(mem_shape, dtype=cplx_type_map[self.param_dtype])
        return h
