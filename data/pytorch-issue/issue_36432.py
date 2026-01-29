# torch.rand(1, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def parse_float_from_str(self, s: str) -> float:
        # Replicates the issue's problematic split-and-parse logic
        parts = torch.split(s, "_", -1)  # Intentionally invalid for demonstration
        first_part = parts[0]
        return float(first_part)

    def forward(self, x):
        # Hardcoded input string to trigger the TorchScript parsing error
        s = "0_1"
        parsed_val = self.parse_float_from_str(s)
        return x * parsed_val  # Dummy computation to retain tensor output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

