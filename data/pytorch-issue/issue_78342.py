# torch.rand(1, 1, 8, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        kernel_size = 2
        stride = 2
        return_indices = False
        self.max_pool = torch.nn.MaxPool1d(kernel_size, stride=stride, return_indices=return_indices)

    def forward(self, x):
        return self.max_pool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 8, dtype=torch.float32)

