import torch

device = "mps"
a = torch.tensor([[]], device=device, requires_grad=True)
b = torch.tensor([[]], device=device, requires_grad=True)
y = torch.sub(a, b)   # torch.add works without issue.
y.sum().backward()  # seg fault