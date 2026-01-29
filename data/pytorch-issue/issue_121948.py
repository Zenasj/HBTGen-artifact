# torch.rand((), dtype=torch.float32)  # Inferred input shape is a scalar (0D)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU()  # PReLU layer with single parameter
        # Initialize weight to infinity as per the issue's scenario
        self.prelu.weight.data.fill_(float('inf'))

    def forward(self, x):
        return self.prelu(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random scalar tensor (0D) to match PReLU's input requirements
    return torch.rand(()).float()

