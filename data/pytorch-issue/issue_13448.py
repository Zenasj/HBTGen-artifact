# torch.rand(10000, dtype=torch.float32)  # Input shape inferred from issue's test cases
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Convert input to 32-bit and 64-bit tensors
        float_input = x.float()
        double_input = x.double()
        
        # Apply sigmoid to both
        sigmoid_float = torch.sigmoid(float_input)
        sigmoid_double = torch.sigmoid(double_input)
        
        # Compute proportion of NaNs in each output
        nan_float = torch.isnan(sigmoid_float).float().mean()
        nan_double = torch.isnan(sigmoid_double).float().mean()
        
        # Return True if proportions differ by more than 1e-5 (captures discrepancies like 0.002 vs 1.0)
        return torch.abs(nan_float - nan_double) > 1e-5

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a tensor of 10000 NaNs (matches issue's largest test case)
    return torch.zeros(10000) / 0  # Division by zero creates NaNs

