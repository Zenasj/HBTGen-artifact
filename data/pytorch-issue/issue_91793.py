# torch.rand(0, 5, 2, 2, 2, dtype=torch.float64) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Using a small number of features to avoid OOM
        self.bn = nn.BatchNorm3d(5, eps=0.001, momentum=0.3, affine=False)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(0, 5, 2, 2, 2, dtype=torch.float64)

