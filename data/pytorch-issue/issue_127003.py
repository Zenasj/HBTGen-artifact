# torch.rand(4, dtype=torch.bool), torch.rand(4, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        try:
            x = x - y  # Intentional unsupported op to trigger Dynamo error
        except:
            return x
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Exact tensors from original repro case
    return (
        torch.tensor([1, 0, 1, 0], dtype=torch.bool),
        torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    )

