import numpy as np

import jax
import jax.numpy as jnp

# Define the conditional branch function using jax.lax.cond                                                                                                                                                 
def cond_branch(x, w1, w2):
    return jax.lax.cond(
        x > 0,
        lambda x: w1 * x,
        lambda x: w2 * x,
        x,
    )

# Inputs                                                                                                                                                                                                    
x = jnp.ones(())
neg_x = -jnp.ones(())
w1 = jnp.zeros(())
w2 = jnp.zeros(())

w_grads = jax.grad(cond_branch, [1, 2])(x, w1, w2)

print("GALVEZ:", w_grads[0], w_grads[1])