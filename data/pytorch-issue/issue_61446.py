# torch.rand(())

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_full_backward_hook(lambda *args: None)

    def forward(self, input):
        return input

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(()).requires_grad_(True)

