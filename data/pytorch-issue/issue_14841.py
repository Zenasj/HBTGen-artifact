# torch.rand(2, 4, 6, 6, dtype=torch.float)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=3, groups=2)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(2, 4, 6, 6, dtype=torch.float, requires_grad=True)

