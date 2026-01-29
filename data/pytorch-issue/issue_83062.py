# torch.rand(16, 960, 4, 4, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using a valid output size for AdaptiveAvgPool2d
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        return self.adaptive_avg_pool(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(16, 960, 4, 4, dtype=torch.float32)

