# torch.rand(B, 6, dtype=torch.float32), classes=torch.tensor([0, 1], dtype=torch.int64)
import torch
from torch import nn
from typing import Optional

class MyModel(nn.Module):
    def forward(self, x: torch.Tensor, classes: Optional[torch.Tensor] = None) -> torch.Tensor:
        if classes is not None:
            # Compare 5th column (index 5) with provided classes
            x_col = x[:, 5].unsqueeze(1)  # (N, 1)
            comparison = (x_col == classes.unsqueeze(0))  # (N, C)
            mask = comparison.any(dim=1)  # (N,)
            x = x[mask]
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 32  # Example batch size
    x = torch.rand(B, 6, dtype=torch.float32)  # 6 columns to match x[:,5] usage
    classes = torch.tensor([0, 2], dtype=torch.int64)  # Example class indices
    return (x, classes)

