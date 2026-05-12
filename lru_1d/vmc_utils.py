import jax
import jax.numpy as jnp
from flax import nnx



def compute_local_energies_scan(model, tfim_J, tfim_h, site_coords, samples, log_psi_original, parallel, unroll=1):
    # Compute the local energy for each sample
    # samples is of shape (Lx, ..., batch_size)
    # log_psi_original is of shape (batch_size,)
    
    spin_samples = 2*samples - 1
    inter_energy = jnp.vecdot(spin_samples[:-1, ...], spin_samples[1:, ...], axis=0, preferred_element_type=log_psi_original.dtype)
        
    @nnx.scan(in_axes=(None, None, 0, nnx.Carry),
              out_axes=nnx.Carry,
              length=len(site_coords),
              unroll=unroll)
    def scan_fn(model, samples, idx, carry):
        idx = tuple(idx)
        samples = samples.at[*idx, ...].set(1 - samples[*idx, ...])
        log_psi_flip = model.log_psis(samples, parallel=parallel)
        samples = samples.at[*idx, ...].set(1 - samples[*idx, ...])
        return (carry + jnp.exp(log_psi_flip - log_psi_original))
    
    field_energy = scan_fn(model, samples, site_coords, jnp.zeros_like(log_psi_original))

    return tfim_J*inter_energy - tfim_h*field_energy


# set clip-rho to 10 by default i.e. basically no clipping
def estimate_energy_and_gradient(model, samples, E_loc, clip_rho=10.0):
    # samples are of shape (Lx, batch_size)
    # log_psi_original is of shape (batch_size,)
    # E_loc is of shape (batch_size,)

    # compute the energy as well
    E_average = jnp.mean(E_loc.real)
    E_variance = jnp.var(E_loc.real)
    E_error = jnp.sqrt(E_variance/E_loc.shape[0])
    E_stats = {"mean": E_average, "error_of_mean": E_error, "variance": E_variance}
    
    E_median = jnp.median(E_loc.real)
    mad = jnp.abs(E_loc - E_median).mean()
    clip_min, clip_max = (E_median - clip_rho*mad), (E_median + clip_rho*mad)
    
    E_loc_clip = jnp.clip(E_loc.real, min=clip_min, max=clip_max)
    E_clip_avg = jnp.mean(E_loc_clip)
    E_clip_var = jnp.var(E_loc_clip)
    E_clip_err = jnp.sqrt(E_clip_var/E_loc_clip.shape[0])
    
    E_stats = {"energy_mean": E_average, "energy_error_of_mean": E_error, "energy_variance": E_variance,
               "clip_energy_mean": E_clip_avg, "clip_energy_error_of_mean": E_clip_err, "clip_energy_variance": E_clip_var,
               "energy_median": E_median, "energy_MAD_from_median": mad, "energy_clip_min": clip_min, "energy_clip_max": clip_max}
    E_shifted = E_loc_clip - E_clip_avg
    
    def loss_fn(model):
        log_probs = model.log_probs(samples, parallel=False)
        
        losses = jnp.real(E_shifted * log_probs)
        losses_mean = jnp.mean(losses)
        losses_var = jnp.var(losses)
        losses_err = jnp.sqrt(losses_var / E_shifted.shape[0])
        
        return losses_mean, {"loss_mean": losses_mean, "loss_median": jnp.median(losses), 
                             "loss_variance": losses_var, "loss_error_of_mean": losses_err}

    # compute the gradient
    (loss, loss_stats), grad = nnx.value_and_grad(loss_fn, has_aux=True)(model)

    stats = {**E_stats, **loss_stats}
    return stats, loss, grad



@nnx.jit(donate_argnames=["model", "optimizer"], static_argnames=["numsamples", "nx", "parallel", "clip_rho"])
def train_step(model, optimizer, rngs, tfim_J, tfim_h, site_coords, numsamples, nx, parallel, clip_rho):
    samples, clp = model.sample((nx,), numsamples, rngs=rngs)
    log_psi_original = model.log_psis(samples, clp)
    
    E_loc = compute_local_energies_scan(model, tfim_J, tfim_h, site_coords, samples, log_psi_original, parallel=parallel)

    stats, loss, grad = estimate_energy_and_gradient(model, samples, E_loc, clip_rho=clip_rho)
    optimizer.update(model, grad)
    return model, optimizer, stats, loss
