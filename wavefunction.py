"""Implements the NCS wavefunction and utilities for its optimization."""

from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim

@partial(jax.jit, static_argnums=1)
def psi(variational_pars, model, sample):
  """Computes the NCS variational wavefunction."""
  psi_raw = model.apply({'params': variational_pars}, sample)
  return jnp.prod(psi_raw, axis=-1)

class Network(nn.Module):
  """Implements the network architecture.
  
  Includes the multilayer perceptron (MLP) and the coherent-state-inspired
  transformation of the MLP output. Multiplication of outputs is performed
  in psi() function in this module, not in the __call__() function here.
  """
  
  num_k: int
  hidden: int
  @nn.compact
  def __call__(self, state):
    x = nn.Dense(features=self.hidden)(state)
    x = nn.tanh(x)
    # More layers possible, e.g.:
    # x = nn.Dense(features=64)(x)
    # x = nn.tanh(x)
    x = nn.Dense(features=self.num_k)(x)
    coherent_num = jnp.power(x, state)
    coherent_den = jnp.sqrt(jnp.exp(jax.scipy.special.gammaln(state + 1.)))
    out = coherent_num/coherent_den
    return out

def get_initial_params(key, shape, model):
  """Creates initial variational parameters."""
  init_val = jnp.ones(shape, jnp.float32)
  initial_params = model.init(key, init_val)['params']
  return initial_params

def create_optimizer(params, learning_rate):
  """Creates the optimizer.
  
  Optimizer is a Flax object handling optimization of the variational
  parameters.
  """
  optimizer_def = optim.Adam(learning_rate=learning_rate)
  optimizer = optimizer_def.create(params)
  return optimizer
