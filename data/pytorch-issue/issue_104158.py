import torch
x = torch.rand(10, 10)
x_inv = torch.linalg.inv(x)
print(((x_inv @ x) - torch.eye(10)).abs().max())