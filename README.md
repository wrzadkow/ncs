# Neural coherent states

Implementation of neural coherent states, a type of neural-network quantum
states introduced in the following preprint:

Artificial neural network states for non-additive systems  
Wojciech Rzadkowski, Mikhail Lemeshko, Johan H. Mentink  
[arXiv:2105.15193](https://arxiv.org/abs/2105.15193)

## Dependencies

This code needs [Jax](https://github.com/google/jax) and 
[Flax](https://github.com/google/flax).
Python 3.9 is recommended.

## Using the code
Running `python main.py` will perform learning procedure for a small system
 with two bosonic modes. Energies at each optimization step will be written to `output.txt`. For
simplicity, adjusting both the physics and algorithm parameters is done [directly
in the main file](https://github.com/wrzadkow/ncs/blob/main/main.py#L18) by editing
 `physics_pars` and `arg_pars` variables.

The code runs on GPU without change. Consult [this material](https://github.com/google/jax/blob/master/cloud_tpu_colabs/README.md#running-jax-on-a-cloud-tpu-from-a-gce-vm) for running on TPU.

The tests can be run with `python tests.py`. No errors indicate tests passing, 
while `AssertionError`s appearing correspond to their failure.





