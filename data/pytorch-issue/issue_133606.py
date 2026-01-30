import torch

3
def f(x):
    # Compute trace
    mask = torch.eye(2)
    I = (x * mask[None, ...]).sum(axis=(1, 2))
    # Compute determinant
    J = torch.linalg.det(x)
    # Compute f
    return I + J**0.5

torch.autograd.gradcheck(f, torch.rand(9, 2, 2, dtype=torch.double, requires_grad=True))