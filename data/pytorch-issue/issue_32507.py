# torch.rand(25, 1, 240640, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_dim=256, kernel_size=769, stride=64, padding=0, padding_mode='zeros'):
        super(MyModel, self).__init__()
        self.conv1d = nn.Conv1d(
            1,
            hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x):
        return self.conv1d(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn([25, 1, 240640])

