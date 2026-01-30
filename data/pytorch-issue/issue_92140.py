import numpy as np

import torch
import functorch
import jax.numpy as jnp
import jax

x = torch.tensor(0.5 + 0.5j, dtype=torch.complex128)

def fn(x):
    return x.sin()

jacfwd_o = functorch.jacfwd(fn)(x)
jacrev_o = functorch.jacrev(fn)(x)
jacobian_fwd_o = torch.autograd.functional.jacobian(fn, x, vectorize=True, strategy='forward-mode')
jacobian_rev_o = torch.autograd.functional.jacobian(fn, x)

print("jacfwd:", jacfwd_o)
print("functional.jacobian (fwd):", jacobian_fwd_o)

print("jacrev:", jacrev_o)
print("functional.jacobian (rev):", jacobian_rev_o)

print("***"*5, "JAX", "***"*5)

def jax_fn(x):
    return jnp.sin(x)

jax_fwd = jax.jacfwd(jax_fn, holomorphic=True)(x.numpy())
jax_rev = jax.jacrev(jax_fn, holomorphic=True)(x.numpy())
print("jax.jacfwd", jax_fwd)
print("jax.jacrev", jax_rev)