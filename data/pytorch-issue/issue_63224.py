# torch.rand(1, 3, 2, 1, dtype=torch.float)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    x = torch.arange(6, dtype=torch.float).reshape(1, 3, 2, 1)
    x = x.permute(0, 2, 1, 3)
    return x

