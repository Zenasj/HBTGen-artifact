# torch.rand(1, 1, 10, 10, dtype=torch.float32)  # Inferred input shape based on test examples
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old_division = OldDivision()
        self.new_division = NewDivision()
    
    def forward(self, inputs):
        a, b = inputs
        old_out = self.old_division(a, b)
        new_out = self.new_division(a, b)
        
        # Compute differences between old and new outputs
        nan_diff = torch.isnan(old_out) != torch.isnan(new_out)
        val_diff = ~torch.isclose(old_out, new_out, equal_nan=False)
        return torch.any(nan_diff | val_diff)  # Return True if any differences exist

class OldDivision(nn.Module):
    """Simulates pre-Numpy 1.20 behavior (NaN instead of Inf for division by zero)"""
    def forward(self, a, b):
        result = torch.floor_divide(a, b)
        # Replace Infs with NaNs to match old numpy behavior
        result[torch.isinf(result)] = float('nan')
        return result

class NewDivision(nn.Module):
    """Represents post-Numpy 1.20 behavior (returns Inf for division by zero)"""
    def forward(self, a, b):
        return torch.floor_divide(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 1, 10, 10  # Matches test case dimensions
    a = torch.rand(B, C, H, W, dtype=torch.float32) * 10 - 5  # Random values between -5 and +5
    b = torch.randint(-5, 6, (B, C, H, W), dtype=torch.float32)  # Includes zeros/negatives
    # Explicitly set a division-by-zero case (as seen in test expectations)
    b[0,0,2,0] = 0  # Trigger division by zero at position (2,0)
    return (a, b)  # Returns tuple of dividend and divisor tensors

