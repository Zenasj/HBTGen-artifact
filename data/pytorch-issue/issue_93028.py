py
import torch
from torch.autograd.functional import jacobian
x = torch.zeros(3,3, dtype=torch.complex128)

def func(x):
    x = torch.fft.ifft(x)
    return x

jac_rev = jacobian(func, (x.clone().requires_grad_(), ), strategy='reverse-mode', vectorize=True)[0][0]
jac_fwd = jacobian(func, (x.clone().requires_grad_(), ), strategy='forward-mode', vectorize=True)[0][0]
print(torch.isclose(jac_rev, jac_fwd, atol=1e-4, rtol=1e-4))

tensor([[[ True,  True,  True],
         [ True,  True,  True],
         [ True,  True,  True]],

        [[ True, False, False],
         [ True,  True,  True],
         [ True,  True,  True]],

        [[ True, False, False],
         [ True,  True,  True],
         [ True,  True,  True]]])

py
a = torch.tensor([1], dtype=torch.complex64)

def func(a):
    b = a * 1j
    return b

jac_rev = jacobian(func, (a.clone().requires_grad_(), ), strategy='reverse-mode', vectorize=True)[0][0]
jac_fwd = jacobian(func, (a.clone().requires_grad_(), ), strategy='forward-mode', vectorize=True)[0][0]
print(torch.isclose(jac_rev, jac_fwd, atol=1e-4, rtol=1e-4))

print(jac_rev)
print(jac_fwd)