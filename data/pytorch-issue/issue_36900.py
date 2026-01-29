# torch.rand(100000000, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No parameters needed for this model

    def forward(self, x):
        # Use the workaround for max over a dimension
        ii = x.argmax(dim=0)
        maxval = x.gather(0, ii.unsqueeze(0)).squeeze(0)
        return maxval

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(int(100e6), dtype=torch.float32).cuda()

