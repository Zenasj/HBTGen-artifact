import random

g_x = (grad.conj() * y * pow(x, y - 1)).conj()
g_y =  (grad.conj() * pow(x, y) * log(x)).conj()

g_x = (grad.conj() * y * pow(x, y - 1)).conj()
g_y =  (grad.conj() * pow(x, y) * log(x.astype(y.dtype))).conj()

import jax
import numpy as np
import torch
np.random.seed(42)


x = np.random.randn(3)
y = np.random.randn(3) + 1j * np.random.randn(3)
g = np.random.randn(3) + 1j * np.random.randn(3)

print("x: ", x)
print("y: ", y)
print("============================")

device = torch.device("cpu:0")
x1 = torch.tensor(x, device=device, requires_grad=True)
y1 = torch.tensor(y, device=device, requires_grad=True)
g1 = torch.tensor(g, device=device)
o1 = torch.pow(x1, y1)
o1.backward(g1)
print("o1: \n", o1.detach().cpu().numpy())
print("x1g: \n", x1.grad.detach().cpu().numpy())
print("y1g: \n", y1.grad.detach().cpu().numpy())


print("============================")
o2, f_vjp = jax.vjp(jax.numpy.power, x, y)
x2g, y2g = f_vjp(g.conj()) # note that Jax use a different convention for complex gradient
print("o2: \n", np.asarray(o2))
print("x2g: \n", np.asarray(x2g))
print("y2g: \n", np.asarray(y2g))