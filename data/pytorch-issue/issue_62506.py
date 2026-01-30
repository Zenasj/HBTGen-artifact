import torch

x = torch.tensor([[1, 2], [3, 4.]], requires_grad=True)
torch.isin(x, torch.tensor([2, 3]))