# torch.rand(1, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        cond = x[0].item()
        if cond:
            return torch.tensor(1, dtype=torch.int32)  # Returns int tensor
        else:
            return torch.tensor(1.0, dtype=torch.float32)  # Returns float tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a boolean tensor to trigger the conditional return in forward()
    return torch.tensor([True], dtype=torch.bool)

