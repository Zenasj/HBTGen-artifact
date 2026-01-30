import torch

x = torch.rand(4, 1, 224, 224)
dy, dx = torch.gradient(x, dim=(2,3), edge_order=1)