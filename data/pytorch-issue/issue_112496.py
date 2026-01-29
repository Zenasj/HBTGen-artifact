# torch.rand(9, 10, 9, 8, 6, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x, _ = torch.nn.functional.adaptive_max_pool3d_with_indices(x, output_size=(5, 5, 5), return_indices=True)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(9, 10, 9, 8, 6, dtype=torch.float32)

