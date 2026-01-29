# torch.rand(2, 3, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, permutation):
        super(MyModel, self).__init__()
        self.permutation = permutation

    def forward(self, x):
        dims = x.dim()
        if len(self.permutation) != dims:
            raise RuntimeError("number of dims don't match in permute")
        if len(set(self.permutation)) != dims:
            raise RuntimeError("repeated dim in permute")
        return x.permute(self.permutation)

def my_model_function():
    # Example permutation (3,2,1,0) for 4D tensors as in the debug traces
    return MyModel((3, 2, 1, 0))

def GetInput():
    # Generates a 4D tensor with shape (2,3,2,2) as in test cases
    return torch.rand(2, 3, 2, 2, dtype=torch.float32)

