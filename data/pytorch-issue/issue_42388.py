# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

MY_STR_CONST = "Hi, I am a string, please realize I am a constant"

class MyModel(nn.Module):
    def forward(self, x):
        # This usage of a global constant MY_STR_CONST will trigger the TorchScript error
        if MY_STR_CONST == "Hi, I am a string, please realize I am a constant":
            return x + 1  # Dummy computation to return a tensor
        else:
            return x - 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

