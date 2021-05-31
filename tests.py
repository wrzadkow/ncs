"""Tests crucial components."""

import jax
import jax.numpy as jnp

from main import PhysicsParameters
from energy import local_energy, H_finite_mass
from wavefunction import Network, psi, create_optimizer, get_initial_params

def test_local_energy():
  """Tests local energy.
  
  This requires initializing the model – this is hence also tested.
  """
  test_acc = 0.00001
  pars = PhysicsParameters(
      V=0.1, inv_mass=0., k_grid=jnp.array([-1., 1.]), n_max=3)
  state = jnp.array([0, 0])
  model = Network(num_k=2, hidden=32)
  key = jax.random.PRNGKey(0)
  shape = (2, )
  init_variational_pars = get_initial_params(key, shape, model)
  optimizer = create_optimizer(init_variational_pars, 3e-4)
  le = local_energy(state, model, optimizer.target, pars)
  states = jnp.array([[0, 0], [1, 0], [0, 1]])
  psi_val = psi(init_variational_pars, model, states)
  H0 = H_finite_mass(state, jnp.array([0,0]), pars)
  H1 = H_finite_mass(state, jnp.array([1,0]), pars)
  H2 = H_finite_mass(state, jnp.array([0,1]), pars)
  target_value = (psi_val[1] * H1 + psi_val[2] * H2 + psi_val[0] * H0)/psi_val[0]
  assert abs(le - target_value) < test_acc

def test_H_finite_mass():
  """Tests implementation of the Hamiltonian."""
  test_acc = 0.00001

  pars = PhysicsParameters(
      V=0.05, inv_mass=0.5, k_grid=jnp.array([-1., 0., 1.]), n_max=3)
  x = jnp.array([1, 1, 2])
  y = jnp.array([1, 1, 2])
  actual_val  = H_finite_mass(x, y, pars)
  true_val = 0.5 * ((-1. * 1 + 1. * 2) ** 2) + 4.
  assert abs(actual_val - true_val) <= test_acc

  pars = PhysicsParameters(
      V=0.05, inv_mass=0.5, k_grid=jnp.array([-1., 0., 1.]), n_max=3)
  x = jnp.array([1, 0, 0])
  y = jnp.array([0, 0, 0])
  actual_val_2  = H_finite_mass(x, y, pars)
  actual_val_1  = H_finite_mass(y, x, pars)
  true_val = pars.V
  assert abs(actual_val_1 - true_val) <= test_acc
  assert abs(actual_val_2 - true_val) <= test_acc
  
  pars = PhysicsParameters(
      V=0.05, inv_mass=0.5, k_grid=jnp.array([-0.3, 0.2, 0.8]), n_max=3)
  x = jnp.array([3, 3, 2])
  y = jnp.array([3, 3, 2])
  actual_val  = H_finite_mass(x, y, pars)
  true_val = pars.inv_mass * jnp.square(jnp.sum(jnp.multiply(pars.k_grid, x)))
  true_val += jnp.sum(x)
  assert abs(actual_val - true_val) <= test_acc


  pars = PhysicsParameters(
      V=0.05, inv_mass=0.5, k_grid=jnp.array([-0.3, 0.2, 0.8]), n_max=3)
  x = jnp.array([3, 5, 2])
  y = jnp.array([3, 3, 2])
  actual_val  = H_finite_mass(x, y, pars)
  true_val = 0.
  assert abs(actual_val - true_val) <= test_acc

  pars = PhysicsParameters(
      V=0.05, inv_mass=0.5, k_grid=jnp.array([-0.3, 0.2, 0.8]), n_max=3)
  x = jnp.array([2, 4, 2])
  y = jnp.array([3, 4, 2])
  actual_val  = H_finite_mass(x, y, pars)
  true_val = jnp.sqrt(3.) * pars.V
  assert abs(actual_val - true_val) <= test_acc

test_local_energy()
test_H_finite_mass()
