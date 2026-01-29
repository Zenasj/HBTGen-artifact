# Input is a tuple of two sparse COO tensors (self_tensor, other_tensor) with compatible dimensions
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        self_tensor, other_tensor = inputs
        indices = other_tensor._indices()
        values = torch.zeros_like(other_tensor._values())
        new_size = other_tensor.size()
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=new_size
        ).coalesce()

def my_model_function():
    return MyModel()

def GetInput():
    # Create two sparse tensors as in the issue's example
    indices1 = torch.zeros((1, 0), dtype=torch.int64)
    values1 = torch.zeros((0,))
    s1 = torch.sparse_coo_tensor(
        indices=indices1,
        values=values1,
        size=(3,)
    ).coalesce()

    indices2 = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    values2 = torch.ones((3,))
    s2 = torch.sparse_coo_tensor(
        indices=indices2,
        values=values2,
        size=(3,)
    ).coalesce()
    return (s1, s2)

