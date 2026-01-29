# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu_model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.gelu_model = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        relu_output = self.relu_model(x)
        gelu_output = self.gelu_model(x)
        return relu_output, gelu_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 1, 8, 8  # Example batch size, channels, height, and width
    return torch.rand(B, C, H, W, dtype=torch.float32)

