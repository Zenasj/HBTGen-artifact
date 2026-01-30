import torch

x = torch.nested_tensor([torch.randn(2, 3), torch.randn(2, 4)])
print(x)