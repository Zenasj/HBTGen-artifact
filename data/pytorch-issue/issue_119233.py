import torch

x = torch.randn(5)
print(x.storage().resizable())  # True
x.numpy()
print(x.storage().resizable())  # False