# torch.rand(1, 1, 5, 5, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), dtype=dtype)

    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(dtype=torch.float32)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 1, 5, 5, dtype=torch.float32)

