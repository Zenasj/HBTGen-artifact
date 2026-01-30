import torch
torch.set_num_threads(1)  # optional
x = torch.randn(10, requires_grad=True)
x.sum().backward()