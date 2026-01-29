# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.prelu = nn.PReLU(16)
    
    def forward(self, x):
        return self.prelu(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    shape = (2, 16, 96, 96)  # (batch_size, channels, height, width)
    return torch.randn(*shape, dtype=torch.float32)

