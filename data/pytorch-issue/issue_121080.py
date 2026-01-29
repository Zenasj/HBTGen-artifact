import torch
from torch import nn

# torch.rand(B, dtype=torch.int64)  # Input shape: 1D tensor of integers
class MyModel(nn.Module):
    def __init__(self, dtype=torch.int64):
        super().__init__()
        self.dtype = dtype

    def forward(self, indptr):
        nodes = torch.arange(len(indptr) - 1, dtype=self.dtype, device=indptr.device)
        return nodes.to(self.dtype).repeat_interleave(indptr.diff())

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a sample indptr tensor with increasing integer values
    B = torch.randint(5, 10, (1,)).item()  # Random length between 5-10 for testing
    indptr = torch.cat([torch.tensor([0], dtype=torch.int64), torch.cumsum(torch.randint(1, 5, (B,)), 0)])
    return indptr

