import torch
device = torch.device("cuda:0")
dtype = torch.double

X = torch.rand(25, 2, device=device, dtype=dtype)
X.requires_grad_(True)

def test_grads(x):
    return torch.cdist(x, x).sum()

torch.autograd.gradcheck(test_grads, (X,))

# test cpu
X_cpu = X.detach().clone().cpu()
X_cpu.requires_grad_(True)
torch.autograd.gradcheck(test_grads, (X_cpu,))

True

import torch
device = torch.device("cuda:0")
dtype = torch.double

X = torch.rand(3, 5, 2, device=device, dtype=dtype)
X.requires_grad_(True)

def test_grads(x):
    return torch.cdist(x, x).sum()

torch.autograd.gradcheck(test_grads, (X,))

True