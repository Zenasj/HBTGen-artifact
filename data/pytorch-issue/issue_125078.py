import torch
from torch.func import jacfwd

def func(x):
    two = 2.0
    return two * x


def jac_func(x):
    return jacfwd(func, argnums=(0,))(x)


compiled_jac_func = torch.compile(jac_func)
compiled_jac_func(torch.ones((3,), dtype=torch.float64))

import torch
from torch.func import jacfwd

def func(x):
    two = 2.0
    return two * x


def jac_func(x):
    return jacfwd(func, argnums=(0,))(x)


compiled_jac_func = torch.compile(jac_func)
y = compiled_jac_func(torch.ones((3,), dtype=torch.float64))
print(y)
# (tensor([[2., 0., 0.],
#         [0., 2., 0.],
#         [0., 0., 2.]], dtype=torch.float64),)