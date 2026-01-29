# torch.rand(1, dtype=torch.float64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        part1 = torch.exp(x ** 2)
        part2 = torch.special.erfc(x)
        expected = part1 * part2  # Expected NaN when x is inf
        actual = torch.special.erfcx(x)  # Actual output (reported as 0 when x is inf)
        # Check if outputs differ, considering NaN vs non-NaN cases
        return torch.tensor([not torch.allclose(actual, expected, equal_nan=False)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input that triggers the issue (infinity)
    return torch.tensor([float('inf')], dtype=torch.float64)

