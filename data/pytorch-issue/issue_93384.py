# torch.rand(2, 2, 3, 6, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        o0, o1 = torch.nn.functional.max_pool2d_with_indices(
            x, 
            kernel_size=[3, 2], 
            stride=[2, 1], 
            padding=[1, 1], 
            dilation=[1, 2], 
            ceil_mode=False, 
            return_indices=True
        )
        return o0

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 2, 3, 6, dtype=torch.float32)

