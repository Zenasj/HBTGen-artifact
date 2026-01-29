# torch.rand(1, 3, 300, 300, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
from collections import namedtuple

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Example: Using a simple convolutional layer for demonstration
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        # Return a namedtuple
        NT = namedtuple('Output', ['attribute_1', 'attribute_2'])
        return NT(x, x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 300, 300, dtype=torch.float32)

