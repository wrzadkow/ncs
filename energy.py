"""Implements the Hamiltonian, calculations of energy and its gradients."""
from functools import partial

import jax
import jax.numpy as jnp

from sampler import connected_states
from wavefunction import psi

@jax.jit
def H_infinite_mass(np, n, physics_pars):
  """Calculates the inifine mass Hamiltonian."""
  firstterm = n * jnp.array(np == n, float)
  secondterm = jnp.sqrt(n) * jnp.array(np == n - 1, float)
  secondterm += jnp.sqrt(np) * jnp.array(np - 1 == n, float)
  return firstterm + physics_pars.V * secondterm

@jax.jit
def H_finite_mass(np, n, physics_pars):
  """Calculates the finite mass Hamiltonian H_{n, np}.
  
  First map the infinite mass Hamiltonian over the bosonic modes using the fact
  that infinite mass Hamiltonian is separable. Then compute and add the
  finite mass term."""
  mapped_infinite = jax.vmap(H_infinite_mass, in_axes=(0,0,None))
  infinite_components = mapped_infinite(np, n, physics_pars)
  equals = jnp.array(np == n)
  # equals_but_one is equivalent to this code:
  # equals_but_one = [jnp.all(equals[:i]) and jnp.all(equals[(i+1):])
  # for i in range(len(equals))]
  equals_but_one = jnp.logical_or(jnp.eye(np.shape[0], dtype=bool), equals)
  equals_but_one = jnp.all(equals_but_one, axis=1)
  infinite_term = jnp.sum(jnp.multiply(equals_but_one, infinite_components))
  k = physics_pars.k_grid
  equals = equals.astype(float)
  finite_components = jnp.multiply(np, equals)
  finite_components = jnp.multiply(k, finite_components)
  finite_term = jnp.square(jnp.prod(equals) * jnp.sum(finite_components))
  finite_term *= physics_pars.inv_mass
  return infinite_term + finite_term

@partial(jax.jit, static_argnums=1)
def local_energy(state, model, variational_pars, physics_pars):
  """Computes local energy given a state."""
  conn, physical = connected_states(state, physics_pars.n_max)
  physical = physical.astype(float)
  psi_this = psi(variational_pars, model, state)
  psi_conn = psi(variational_pars, model, conn)
  me = jax.vmap(H_finite_mass, in_axes=(None,0,None))(state, conn, physics_pars)
  prods = jnp.multiply(me, psi_conn)
  prods = jnp.multiply(prods, physical)
  prods = prods / psi_this
  return jnp.sum(prods)

@partial(jax.jit, static_argnums=1)
def log_grad_psi(variational_pars, model, physics_pars, sample):
  """Calculates the logarithmic derivative of the wavefunction."""
  psi_value, psi_grad = jax.value_and_grad(psi)(
      variational_pars, model, sample)
  return jax.tree_map(lambda x: x/psi_value, psi_grad)

@partial(jax.jit, static_argnums=1)
def energy_forces(variational_pars, model, physics_pars, samples):
  """Estimates energy and its gradients (forces)."""
  mapped_le = jax.vmap(local_energy, in_axes=(0, None, None, None))
  local_energies = mapped_le(samples, model, variational_pars, physics_pars)
  energy = jnp.mean(local_energies)
  mapped_grad = jax.vmap(log_grad_psi, in_axes=(None, None, None, 0))
  gradients = mapped_grad(variational_pars, model, physics_pars, samples)
  mean_gradient = jax.tree_map(lambda x: jnp.mean(x, axis=0), gradients)
  local_energies = local_energies[:, jnp.newaxis]
  # Calculate pytree of gradients multiplied by respective local energies.
  gradients_x_le = jax.tree_map(
      lambda x: jax.vmap(lambda y, z: y * z, in_axes=(0, 0))(local_energies, x),
      gradients)
  average_prod = jax.tree_map(lambda x: jnp.mean(x, axis=0), gradients_x_le)
  prod_averages = jax.tree_map(lambda x: energy * x, mean_gradient)
  forces = jax.tree_multimap(lambda x, y: x - y, average_prod, prod_averages)
  return energy, forces
