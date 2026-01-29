# torch.rand(1, 2048) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(2048, 2048)
        self.leaky_relu = nn.LeakyReLU(negative_slope=1)

    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.linear(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 2048))

