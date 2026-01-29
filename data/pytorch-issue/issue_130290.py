# torch.randint(0, 10, (6,), dtype=torch.long), torch.randint(0, 10, (5,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        tensor, mapping = inputs
        xx, yy = torch.meshgrid(mapping, tensor, indexing="ij")
        condition = (xx == yy)
        indices = torch.argwhere(condition)
        mapped_values = torch.zeros_like(tensor)
        mapped_values[indices[:, 1]] = indices[:, 0]
        return mapped_values

def my_model_function():
    return MyModel()

def GetInput():
    tensor = torch.randint(0, 10, (6,), dtype=torch.long)
    mapping = torch.randint(0, 10, (5,), dtype=torch.long)
    return (tensor, mapping)

