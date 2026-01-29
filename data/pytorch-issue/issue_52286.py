# torch.rand(B, 1000, 3, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        idx0 = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
        pair = x.index_select(1, idx0.flatten()).view(x.shape[0], 2, -1, 3)
        dist = (pair[:, 0, ...] - pair[:, 1, ...]).norm(2, -1)
        idx1, idx2 = (dist <= 5.0).nonzero().unbind(1)
        idx1 = (idx0[:, idx2] + idx1 * x.shape[1])
        return idx1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1000, 3, 1, dtype=torch.float32, device='cuda') * 5.0

