# torch.tensor([0.+0.3071j], dtype=torch.complex64, requires_grad=True) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute the angle of the complex tensor
        return torch.angle(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor([0. + 0.3071j], dtype=torch.complex64, requires_grad=True)

