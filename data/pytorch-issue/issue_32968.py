# torch.rand(1, 10, 120, 120, dtype=torch.float32)  # Inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(10, 10, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        # Create upper triangular mask to avoid NaNs from -inf * 0
        mask = torch.triu(torch.ones(x.shape, dtype=torch.bool, device=x.device))
        x = x.masked_fill(~mask, 0.0)  # Fill lower triangle with 0
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, 120, 120, dtype=torch.float32)

