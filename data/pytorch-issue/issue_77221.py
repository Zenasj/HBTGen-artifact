import torch
from torch.autograd import gradcheck

def f(a):
    return torch.div(a, 0)

a = torch.rand([], requires_grad=True, dtype=torch.float64)
gradcheck(f, (a))
# GradcheckError: Jacobian mismatch for output 0 with respect to input 0,
# numerical:tensor([[nan]], dtype=torch.float64)
# analytical:tensor([[inf]], dtype=torch.float64)