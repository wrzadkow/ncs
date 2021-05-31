from collections import namedtuple

import jax
import jax.numpy as jnp

from energy import energy_forces
from sampler import generate_samples
from wavefunction import Network, create_optimizer, get_initial_params

PhysicsParameters = namedtuple(
    'PhysicsParameters', ['V', 'inv_mass', 'k_grid', 'n_max'])
AlgorithmParameters = namedtuple(
    'AlgorithmParameters',
    ['num_chains', 'samples_per_chain', 'num_steps', 'burnin', 
     'hidden', 'learning_rate'])

def main():
  phys_pars = PhysicsParameters(
      V=0.4, inv_mass=2., k_grid=jnp.array([-1., 1.]), n_max=4)
  alg_pars = AlgorithmParameters(
      num_chains=5, samples_per_chain=4000, num_steps=400,
      burnin=400, hidden=1024, learning_rate=3.e-4)

  num_k = phys_pars.k_grid.shape[0]
  model = Network(num_k=num_k, hidden=alg_pars.hidden)

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  shape = (num_k, )
  init_variational_pars = get_initial_params(subkey, shape, model)
  optimizer = create_optimizer(init_variational_pars, alg_pars.learning_rate)
  energy_vals = []

  for i in range(alg_pars.num_steps):
    if i % 100 == 0:
      print(f'Completed {i} steps')
    key, subkey = jax.random.split(subkey)
    spl = generate_samples(subkey, model, optimizer.target, alg_pars, phys_pars)
    energy_val, grad = energy_forces(optimizer.target, model, phys_pars, spl)
    optimizer = optimizer.apply_gradient(grad)
    energy_vals.append(energy_val)
    print(energy_val)

  with open('results.txt', 'w') as f:
    for e in energy_vals:
      f.write(f'{e}\n')

if __name__ == "__main__":
   main()
