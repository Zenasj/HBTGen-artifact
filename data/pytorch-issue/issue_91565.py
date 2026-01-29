# torch.rand(B, 4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        indices = torch.tensor([[0, 1, 2], [0, 1, 2]], device=x.device)
        values = torch.tensor([1, 2, 3], device=x.device)
        sparse = torch.sparse_coo_tensor(indices, values, (4, 4), dtype=torch.float32)
        dense = sparse.to_dense()
        return x + dense

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 4, 4, dtype=torch.float32)

