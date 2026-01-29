# torch.rand(B, 3, 9, 9, dtype=torch.float32)  # Input shape inferred from 9x9 screen (RGB channels)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal structure to avoid empty model (required for optimizer)
        self.fc = nn.Linear(3*9*9, 10)  # 3 channels × 9×9 pixels → 10 outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input tensor
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random tensor matching expected input shape (B=1, RGB, 9x9)
    return torch.rand(1, 3, 9, 9, dtype=torch.float32)

