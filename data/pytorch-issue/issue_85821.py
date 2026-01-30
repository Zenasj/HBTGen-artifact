py
import torch

def f(A):
    return torch.linalg.norm(A, ord=1.0)

A = torch.tensor([1.3684], dtype=torch.float32)

print(f(A))

from torch.autograd.functional import jacobian

A1 = A.clone().requires_grad_()
jac1 = jacobian(f, A1, vectorize=True, strategy='forward-mode')

import torch
# functorch is included in the pytorch nightly
from functorch import jacrev 

def f(A):
    return torch.linalg.norm(A, ord=1.0)

A = torch.tensor([1.3684], dtype=torch.float32)
print(f(A))

jacrev(f)(A)