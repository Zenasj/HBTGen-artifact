# torch.rand(2, dtype=torch.float64)  # Inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Failing case: upper=True (bool) and training=True (raises error during backward)
        failing_out = F.rrelu(x, upper=True, training=True)
        # Working case 1: training=False (backward works)
        working_out1 = F.rrelu(x, upper=True, training=False)
        # Working case 2: upper as float (0.2) with training=True (backward works)
        working_out2 = F.rrelu(x, upper=0.2, training=True)
        return failing_out, working_out1, working_out2  # Returns all outputs for comparison

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float64, requires_grad=True)

