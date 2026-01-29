# Input is a tuple of two tensors each with shape (2, 2), e.g., (torch.rand(2, 2), torch.rand(2, 2))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        touched_outputs = []
        for v in x:
            y = torch.zeros_like(v)
            touched_output = v * y
            touched_outputs.append(touched_output)
        return x.__class__(touched_outputs)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(2, 2), torch.rand(2, 2))

