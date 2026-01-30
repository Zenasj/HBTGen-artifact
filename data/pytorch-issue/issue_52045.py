import torch

scattering = Scattering2D(2, shape=(8, 8), backend=backend, frontend='torch').double().to(device)
x = torch.rand(2, 1, 8, 8).double().to(device).requires_grad_()
gradcheck(scattering, x, nondet_tol=1e-5)