# torch.rand(1, dtype=torch.float64)  # Inferred input shape for the tensor

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute the log2 of the input tensor
        log2_x = x.log2()
        
        # Round the log2 values
        rounded_log2_x = log2_x.round()
        
        # Compute 2 to the power of the rounded log2 values
        result = 2 ** rounded_log2_x
        
        return result

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor(0.123, dtype=torch.float64)

