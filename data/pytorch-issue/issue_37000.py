import torch

a = torch.rand(10, requires_grad=True)

with torch.no_grad():
    b = a.numpy()
# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.