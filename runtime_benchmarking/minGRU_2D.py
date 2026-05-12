import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from functools import partial

sim_dtype = jnp.float32

parallel_scan = jax.lax.associative_scan


############
# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

def matrix_init(key, shape, dtype=sim_dtype, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization
##########

# def unpatch_array_from_one_hot(patched_array, nx, ny):
#     numsamples, num_classes = patched_array.shape
    
#     indices = jnp.argmax(array, axis=1)  # Shape: (numsamples,)

#     # We use jnp.unpackbits on the flattened binary representation.
#     flattened_binary = jnp.unpackbits(indices.astype(jnp.uint8), axis=-1, bitorder='little')[:, :, :nx * ny]
    
#     patch_array = flattened_binary.reshape(numsamples, nx, ny)

#     decoded_array = patch_array.transpose(0, 2, 1, 3).reshape(Lx, Ly)

#     return original_array



##############

class TwoDFastGRUlayer(nn.Module):
    """
   2D minGRU (Fast GRU) implementation
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions

    def setup(self):

        # Glorot initialized Input/Output projection matrices

        self.dense = nn.Dense(self.d_hidden)
        self.dense_gate = nn.Dense(self.d_hidden)
        self.dense_inputs = nn.Dense(self.d_model)
        self.C = self.param(
            "C",
            jax.nn.initializers.glorot_uniform(),
            (self.d_hidden, self.d_model),
        )
        # self.normalization = nn.LayerNorm()

    def __call__(self, inputs, previous_inputs, previous_hidden_states):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""

        if previous_inputs != None:
            concatenated_hidden_inputs = jnp.concatenate([inputs,previous_inputs,previous_hidden_states], axis = -1)
            concatenated_inputs = jnp.concatenate([inputs,previous_inputs], axis = -1)
        else:
            concatenated_hidden_inputs = jnp.concatenate([inputs,previous_hidden_states], axis = -1)
            concatenated_inputs = inputs
        # concatenated_inputs =  self.normalization(concatenated_inputs)

        gate = jax.vmap(lambda u: self.dense_gate(u) )(concatenated_inputs) 
        gate = jax.nn.sigmoid(gate)
  
        Lambda_elements = gate

        transformed_hidden_inputs = jax.vmap(lambda u: self.dense(u))(concatenated_hidden_inputs) 

        Bu_elements = (1. - gate) * transformed_hidden_inputs
        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h,i: (h @ self.C) + self.dense_inputs(i))(hidden_states, concatenated_inputs)

        return outputs, hidden_states

    def stateful_call(self, inputs, hidden_states):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t] one step at a time"""

        if inputs[1] != None:
            concatenated_hidden_inputs = jnp.concatenate([inputs[0],inputs[1],hidden_states[1]], axis = -1)
            concatenated_inputs = jnp.concatenate([inputs[0],inputs[1]], axis = -1)
        else:
            concatenated_hidden_inputs = jnp.concatenate([inputs[0],hidden_states[1]], axis = -1)
            concatenated_inputs = inputs[0]

        # concatenated_inputs =  self.normalization(concatenated_inputs)

        gate = self.dense_gate(concatenated_inputs)
        gate = jax.nn.sigmoid(gate)

        transformed_hidden_inputs =  self.dense(concatenated_hidden_inputs)

        new_hidden_state = gate*hidden_states[0] + (1.-gate)*transformed_hidden_inputs 
        # Use them to compute the output of the module
        outputs = new_hidden_state @ self.C + self.dense_inputs(concatenated_inputs)
        return outputs, new_hidden_state


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    fastGRUlayer: TwoDFastGRUlayer  # lru module
    d_model: int  # model size

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq = self.fastGRUlayer
        # self.out1 = nn.Dense(self.d_model)
        # self.out2 = nn.Dense(self.d_model)
 
    def __call__(self, inputs, previous_inputs, previous_hidden_states, skip_input):
        x, new_hidden_states = self.seq(inputs, previous_inputs, previous_hidden_states)  # call LRU
        if skip_input == None:
            x = nn.gelu(x)
        else:
            x = nn.gelu(x + skip_input)
        # x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        # x = self.drop(x)
        # if previous_inputs != None:
        #     concatenated_inputs = jnp.concatenate([inputs,previous_inputs], axis = -1)
        #     return self.out3(concatenated_inputs) + x, new_hidden_states  # skip connection
        # else:
        #     return self.out3(inputs) + x, new_hidden_states
        return x, new_hidden_states  #no skip connection

    def stateful_call(self, inputs, hidden_states, skip_input):
        
        x, new_hidden_state = self.seq.stateful_call(inputs, hidden_states)  # call LRU
        if skip_input == None:
            x = nn.gelu(x)
        else:
            x = nn.gelu(x + skip_input)
        # x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        # x = self.drop(x)
        # if inputs[1] != None:
        #     concatenated_inputs = jnp.concatenate([inputs[0],inputs[1]], axis = -1)
        #     return self.out3(concatenated_inputs) + x, new_hidden_state  # skip connection
        # else:
        #     return self.out3(inputs[0]) + x, new_hidden_state
        return x, new_hidden_state  # no skip connection

BatchSequenceLayer = nn.vmap(
    SequenceLayer,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "batch_stats": None, "cache": 0, "prime": None},
    split_rngs={"params": False},
    axis_name="batch",
)

class TwoDFastGRU(nn.Module):
    """Encoder containing several SequenceLayer"""

    d_model: int
    d_hidden: int
    n_layers: int
    patch_x: int
    patch_y: int

    def setup(self):
        self.layers = [
            BatchSequenceLayer(
                fastGRUlayer=TwoDFastGRUlayer(d_model = self.d_model, d_hidden = self.d_hidden),
                d_model=self.d_model,
            )
            for _ in range(self.n_layers)
        ]
        self.inputs_size = self.patch_x*self.patch_y*2
        self.output_size = 2**(self.patch_x*self.patch_y)
        self.decoder = nn.Dense(self.output_size)

    def generate_zigzag_path(self, Nx, Ny):
       return [(i if j % 2 == 0 else Nx - 1 - i, j) for j in range(Ny) for i in range(Nx)]

    def zigzag_indices(self, Nx, Ny):
        ny_seq = jnp.repeat(jnp.arange(Ny, dtype=jnp.int32), Nx)
        i_seq = jnp.tile(jnp.arange(Nx, dtype=jnp.int32), Ny)
        nx_seq = jnp.where((ny_seq % 2) == 0, i_seq, Nx - 1 - i_seq)
        return nx_seq, ny_seq

    def patch_array(self,array, px, py):
        numsamples, Lx, Ly = array.shape

        reshaped = array.reshape(numsamples, Lx // px, px, Ly // py, py)

        patched = reshaped.transpose(0, 1, 3, 2, 4)  # Shape: (numsamples, Lx/px, Ly/py, px, py)

        final_array = patched.reshape(numsamples, Lx // px, Ly // py, px * py)
        
        return final_array

    def binary_to_decimal(self, binary_array):
        return jnp.dot(binary_array, 2 ** jnp.arange(binary_array.shape[-1]))
    
    def decimal_to_binary(self,decimal_batch, n_bits):
        binary_array = (decimal_batch[:, None] >> jnp.arange(n_bits - 1, -1, -1)) & 1
        return jnp.flip(binary_array, axis=-1)
        # return binary_array

    def _call_unoptimized(self, inputs):
        numsamples, old_Nx, old_Ny = inputs.shape
        Nx, Ny = old_Nx // self.patch_x, old_Ny // self.patch_y

        hidden_states = [jnp.zeros((numsamples, Nx, self.d_hidden), dtype=sim_dtype) for _ in range(self.n_layers)]
        previous_inputs2 = jnp.zeros((numsamples, Nx, self.d_model), dtype=sim_dtype)
        cond_log_probs = jnp.zeros((numsamples, Nx, Ny, self.output_size), dtype=sim_dtype)

        patched_inputs = self.patch_array(inputs, self.patch_x, self.patch_y)
        onehot_inputs = jax.nn.one_hot(patched_inputs, num_classes=2).reshape(numsamples, Nx, Ny, self.inputs_size)
        padded_inputs = jnp.pad(onehot_inputs, ((0, 0), (1, 1), (1, 0), (0, 0)))

        for j in range(Ny):
            if j % 2 == 0:
                x = padded_inputs[:, :-2, j + 1]
                previous_inputs = padded_inputs[:, 1:-1, j]

                for layer_index, layer in enumerate(self.layers):
                    if layer_index == 0:
                        x, new_h = layer(x, previous_inputs, hidden_states[layer_index], None)
                    elif layer_index == 1:
                        x, new_h = layer(x, None, hidden_states[layer_index], None)
                    else:
                        x, new_h = layer(x, None, hidden_states[layer_index], hidden_states[layer_index - 1])

                    hidden_states[layer_index] = new_h
                    previous_inputs2 = x
                outputs = x
            else:
                x = padded_inputs[:, ::-1][:, :-2, j + 1]
                previous_inputs = padded_inputs[:, 1:-1, j]
                for layer_index, layer in enumerate(self.layers):
                    if layer_index == 0:
                        x, new_h = layer(x, previous_inputs[:, ::-1], hidden_states[layer_index][:, ::-1], None)
                    elif layer_index == 1:
                        x, new_h = layer(x, None, hidden_states[layer_index][:, ::-1], None)
                    else:
                        x, new_h = layer(
                            x,
                            None,
                            hidden_states[layer_index][:, ::-1],
                            hidden_states[layer_index - 1][:, ::-1],
                        )

                    hidden_states[layer_index] = new_h[:, ::-1]
                    previous_inputs2 = x[:, ::-1]
                outputs = x[:, ::-1]

            logits = nn.log_softmax(self.decoder(outputs), axis=-1)
            cond_log_probs = cond_log_probs.at[:, :, j].set(logits)
            previous_inputs = previous_inputs2

        transformed_inputs = jax.nn.one_hot(self.binary_to_decimal(patched_inputs), num_classes=self.output_size)
        return jnp.sum(cond_log_probs * transformed_inputs, axis=(1, 2, 3))

    def __call__(self, inputs):
        """forward pass of the model"""

        # During init() params are being created, so self.variables["params"] is unavailable.
        if "params" not in self.variables:
            return self._call_unoptimized(inputs)

        numsamples, old_Nx, old_Ny = inputs.shape
        Nx, Ny = old_Nx//self.patch_x, old_Ny//self.patch_y
        params = self.variables["params"]

        hidden_states = jnp.zeros((self.n_layers, numsamples, Nx, self.d_hidden), dtype=sim_dtype)

        patched_inputs = self.patch_array(inputs, self.patch_x, self.patch_y)
        onehot_inputs = jax.nn.one_hot(patched_inputs,num_classes=2).reshape(numsamples, Nx, Ny, self.inputs_size)
        padded_inputs = jnp.pad(onehot_inputs, ((0,0),(1,1),(1,0),(0,0))) #padding value is zero by default

        def scan_step(h_states, j):
            col_next = jax.lax.dynamic_index_in_dim(padded_inputs, j + 1, axis=2, keepdims=False)
            col_prev = jax.lax.dynamic_index_in_dim(padded_inputs, j, axis=2, keepdims=False)
            x_even = col_next[:, :-2, :]
            x_odd = col_next[:, 2:, :][:, ::-1, :]
            prev_lr = col_prev[:, 1:-1, :]
            is_even = (j % 2) == 0

            x = jax.lax.cond(is_even, lambda t: t[0], lambda t: t[1], (x_even, x_odd))
            prev0 = jax.lax.cond(is_even, lambda t: t, lambda t: t[:, ::-1, :], prev_lr)

            prev_layer_h_oriented = None
            for layer_index, layer in enumerate(self.layers):
                h_canon = h_states[layer_index]
                h_oriented = jax.lax.cond(is_even, lambda t: t, lambda t: t[:, ::-1, :], h_canon)

                if layer_index == 0:
                    prev_in = prev0
                    skip_input = None
                elif layer_index == 1:
                    prev_in = None
                    skip_input = None
                else:
                    prev_in = None
                    skip_input = prev_layer_h_oriented

                x, h_new_oriented = layer.apply(
                    {"params": params[f"layers_{layer_index}"]},
                    x,
                    prev_in,
                    h_oriented,
                    skip_input,
                )
                h_new_canon = jax.lax.cond(is_even, lambda t: t, lambda t: t[:, ::-1, :], h_new_oriented)
                h_states = h_states.at[layer_index].set(h_new_canon)
                prev_layer_h_oriented = h_new_oriented

            outputs = jax.lax.cond(is_even, lambda t: t, lambda t: t[:, ::-1, :], x)
            logits = nn.log_softmax(
                self.decoder.apply({"params": params["decoder"]}, outputs),
                axis=-1,
            )
            return h_states, logits

        hidden_states, logits_by_col = jax.lax.scan(
            scan_step,
            hidden_states,
            jnp.arange(Ny, dtype=jnp.int32),
        )
        cond_log_probs = jnp.transpose(logits_by_col, (1, 2, 0, 3))

        transformed_inputs = jax.nn.one_hot(self.binary_to_decimal(patched_inputs), num_classes = self.output_size)
        log_probabilities = jnp.sum(cond_log_probs * transformed_inputs, axis = (1,2,3))
    
        return log_probabilities

    def sequential_call(self, samples):
        """Sequential call of the model"""
        numsamples, old_Nx, old_Ny = samples.shape
        Nx, Ny = old_Nx//self.patch_x, old_Ny//self.patch_y
        params = self.variables["params"]

        hidden_states = jnp.zeros((self.n_layers, Nx + 2, Ny + 2, numsamples, self.d_hidden), dtype=sim_dtype)
        layer_outputs = jnp.zeros((self.n_layers, Nx + 2, Ny + 2, numsamples, self.d_model), dtype=sim_dtype)
        inputs0 = jnp.zeros((Nx + 2, Ny + 2, numsamples, self.inputs_size), dtype=sim_dtype)
        cond_log_probs = jnp.zeros((numsamples,Nx,Ny,self.output_size), dtype = sim_dtype)

        patched_inputs = self.patch_array(samples, self.patch_x, self.patch_y)
        onehot_inputs = jax.nn.one_hot(patched_inputs,num_classes=2).reshape(numsamples, Nx, Ny, self.inputs_size)

        nx_seq, ny_seq = self.zigzag_indices(Nx, Ny)

        def scan_step(carry, pos):
            h_buf, out_buf, in0_buf, cond = carry
            nx, ny = pos

            x_idx = nx + 1
            y_idx = ny + 1
            x_prev_idx = x_idx + jnp.where((ny % 2) == 0, -1, 1)
            y_prev_idx = y_idx - 1

            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = in0_buf[x_prev_idx, y_idx]
                    x2 = in0_buf[x_idx, y_prev_idx]
                    skip_input = None
                else:
                    x1 = out_buf[layer_index - 1, x_idx, y_idx]
                    x2 = None
                    skip_input = None if layer_index == 1 else h_buf[layer_index - 1, x_idx, y_idx]

                h1 = h_buf[layer_index, x_prev_idx, y_idx]
                h2 = h_buf[layer_index, x_idx, y_prev_idx]
                out_val, h_new = layer.apply(
                    {"params": params[f"layers_{layer_index}"]},
                    (x1, x2),
                    (h1, h2),
                    skip_input,
                    method=SequenceLayer.stateful_call,
                )
                out_buf = out_buf.at[layer_index, x_idx, y_idx].set(out_val)
                h_buf = h_buf.at[layer_index, x_idx, y_idx].set(h_new)

            logits = nn.log_softmax(
                self.decoder.apply({"params": params["decoder"]}, out_buf[self.n_layers - 1, x_idx, y_idx]),
                axis=-1,
            )
            cond = cond.at[:, nx, ny].set(logits)
            in0_buf = in0_buf.at[x_idx, y_idx].set(onehot_inputs[:, nx, ny])

            return (h_buf, out_buf, in0_buf, cond), None

        (hidden_states, layer_outputs, inputs0, cond_log_probs), _ = jax.lax.scan(
            scan_step,
            (hidden_states, layer_outputs, inputs0, cond_log_probs),
            (nx_seq, ny_seq),
        )

        transformed_inputs = jax.nn.one_hot(self.binary_to_decimal(patched_inputs), num_classes = self.output_size)
        log_probabilities = jnp.sum(cond_log_probs * transformed_inputs, axis = (1,2,3))

        return log_probabilities

    def sample(self,key,numsamples,old_Nx,old_Ny):
        """Sample from the model for a given system size Nx,Ny and a number of samples `numsamples`"""
        
        Nx, Ny = old_Nx//self.patch_x, old_Ny//self.patch_y
        params = self.variables["params"]

        hidden_states = jnp.zeros((self.n_layers, Nx + 2, Ny + 2, numsamples, self.d_hidden), dtype=sim_dtype)
        layer_outputs = jnp.zeros((self.n_layers, Nx + 2, Ny + 2, numsamples, self.d_model), dtype=sim_dtype)
        inputs0 = jnp.zeros((Nx + 2, Ny + 2, numsamples, self.inputs_size), dtype=sim_dtype)
        sample_bits = jnp.zeros((numsamples, Nx, Ny, self.patch_x * self.patch_y), dtype=jnp.int32)

        nx_seq, ny_seq = self.zigzag_indices(Nx, Ny)

        def scan_step(carry, pos):
            rng, h_buf, out_buf, in0_buf, sampled = carry
            nx, ny = pos

            x_idx = nx + 1
            y_idx = ny + 1
            x_prev_idx = x_idx + jnp.where((ny % 2) == 0, -1, 1)
            y_prev_idx = y_idx - 1

            for layer_index, layer in enumerate(self.layers):
                if layer_index == 0:
                    x1 = in0_buf[x_prev_idx, y_idx]
                    x2 = in0_buf[x_idx, y_prev_idx]
                    skip_input = None
                else:
                    x1 = out_buf[layer_index - 1, x_idx, y_idx]
                    x2 = None
                    skip_input = None if layer_index == 1 else h_buf[layer_index - 1, x_idx, y_idx]

                h1 = h_buf[layer_index, x_prev_idx, y_idx]
                h2 = h_buf[layer_index, x_idx, y_prev_idx]
                out_val, h_new = layer.apply(
                    {"params": params[f"layers_{layer_index}"]},
                    (x1, x2),
                    (h1, h2),
                    skip_input,
                    method=SequenceLayer.stateful_call,
                )
                out_buf = out_buf.at[layer_index, x_idx, y_idx].set(out_val)
                h_buf = h_buf.at[layer_index, x_idx, y_idx].set(h_new)

            rng, k = jax.random.split(rng)
            logits = nn.log_softmax(
                self.decoder.apply({"params": params["decoder"]}, out_buf[self.n_layers - 1, x_idx, y_idx]),
                axis=-1,
            )
            patch_decimal = jax.random.categorical(key=k, logits=logits)
            patch_bits = self.decimal_to_binary(patch_decimal, self.patch_x * self.patch_y)
            sampled = sampled.at[:, nx, ny].set(patch_bits)
            in0_buf = in0_buf.at[x_idx, y_idx].set(
                jax.nn.one_hot(patch_bits, num_classes=2).reshape(numsamples, self.inputs_size)
            )

            return (rng, h_buf, out_buf, in0_buf, sampled), None

        (_, hidden_states, layer_outputs, inputs0, sample_bits), _ = jax.lax.scan(
            scan_step,
            (key, hidden_states, layer_outputs, inputs0, sample_bits),
            (nx_seq, ny_seq),
        )

        samples = sample_bits.reshape(numsamples, Nx, Ny, self.patch_x, self.patch_y)
        decoded_samples = samples.transpose(0, 1, 3, 2, 4).reshape(numsamples, old_Nx, old_Ny)
        
        ## Check if samples follow log_probs distribution?
        return decoded_samples

    def logprobs_fromsymmetrygroup(self, list_samples):
        group_cardinal = len(list_samples)
        numsamples, Nx, Ny = list_samples[0].shape
        
        
        # Reshape and concatenate samples
        list_samples = jnp.reshape(jnp.concatenate(list_samples, axis=0), (-1, Nx, Ny))

        ## Multiple GRU support (not working yet)
        # numgpus = jax.local_device_count()  # Number of available GPUs
        # numsamplespergpu = list_samples.shape[0] // numgpus

        # Define a function for GPU-specific computation
        # @partial(jax.pmap, in_axes=(0, None))
        # def compute_log_probs(samples):
        #     log_prob = self.__call__(samples)
        #     return log_probs 

        # Partition samples across GPUs and compute probabilities and phases
        # gpu_samples = jnp.split(list_samples, numgpus)
        # list_logprobs = compute_log_probs(gpu_samples)

        list_logprobs = self.__call__(list_samples)
        
        # Reshape and combine results
        # list_logprobs = jnp.reshape(jnp.concatenate(list_logprobs, axis=0), (group_cardinal, numsamples)) #only when multiple GPUs are used
        list_logprobs = jnp.reshape(list_logprobs, (group_cardinal, numsamples))
            
        # Compute final log amplitude
        avg_logprobs = jax.scipy.special.logsumexp(list_logprobs, axis=0) - jnp.log(group_cardinal)
        
        return avg_logprobs

    def logprobs_c4vsym(self, samples):
        numsamples, Nx, Ny = samples.shape

        # # # Initialize list_samples with the original sample
        list_samples = [samples]
        
        
        # Apply rotations and reflections
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-1, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-2, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(jnp.rot90(samples.reshape(-1, Nx, Ny, 1), k=-3, axes=(1, 2)).reshape(-1, Nx, Ny))
        list_samples.append(samples[:, ::-1])  # Flip along rows
        list_samples.append(samples[:, :, ::-1])  # Flip along columns
        list_samples.append(jnp.transpose(samples, axes=(0, 2, 1)))  # Transpose samples
        list_samples.append(jnp.transpose(list_samples[2], axes=(0, 2, 1)))  # Transpose the 180-degree rotated samples

        # Call the method to compute the log problems with symmetry
        return self.logprobs_fromsymmetrygroup(list_samples)
        
    # def logprobs_c4vsym(self, samples):
    #     def apply_symmetries(sample):
    #         return jnp.array([
    #             sample,
    #             jnp.rot90(sample, k=-1, axes=(0, 1)),
    #             jnp.rot90(sample, k=-2, axes=(0, 1)),
    #             jnp.rot90(sample, k=-3, axes=(0, 1)),
    #             sample[::-1],  # Flip along rows
    #             sample[:, ::-1],  # Flip along columns
    #             jnp.transpose(sample),  # Transpose
    #             jnp.transpose(jnp.rot90(sample, k=-2, axes=(0, 1)))
    #         ])
        
    #     # Vectorized application of symmetries to all samples
    #     batch_symmetries = jax.vmap(apply_symmetries)(samples)
    #     batch_symmetries = batch_symmetries.reshape(-1, samples.shape[1], samples.shape[2])
        
    #     return self.logprobs_fromsymmetrygroup(batch_symmetries, 8)

    # def logprobs_fromsymmetrygroup(self, list_samples, group_cardinal):
    #     log_probs = self.__call__(list_samples)
    #     log_probs = log_probs.reshape(group_cardinal, -1)
    #     avg_logprobs = jax.scipy.special.logsumexp(log_probs, axis=0) - jnp.log(group_cardinal)
    #     return avg_logprobs
