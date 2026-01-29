# torch.rand(B, 1, 5, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.Conv2d(8, 1, 3, padding=1)
        )
        self.model_b = nn.Conv2d(1, 1, 3, padding=1)
    
    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        has_nan_a = torch.isnan(out_a).any()
        has_nan_b = torch.isnan(out_b).any()
        return has_nan_a & (~has_nan_b)  # Returns a bool tensor indicating the bug condition

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 1, 5, 5, dtype=torch.float32)

