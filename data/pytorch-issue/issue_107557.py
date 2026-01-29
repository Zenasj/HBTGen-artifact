# torch.rand(1, 64, 10, 9, 8, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Initialize AdaptiveMaxPool3d with output size (0, None, None)
        self.adaptive_max_pool_3d = nn.AdaptiveMaxPool3d((0, None, None))

    def forward(self, x):
        return self.adaptive_max_pool_3d(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 64, 10, 9, 8, dtype=torch.float32)

