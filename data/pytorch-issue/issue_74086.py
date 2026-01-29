# torch.rand(1, 128, 1024, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        for _ in range(8):
            x = x[:, :, ::2]
        for _ in range(8):
            x = x.repeat_interleave(2, -1)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 128, 1024, dtype=torch.float32)

