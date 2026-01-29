# torch.rand(B, C, L, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)  # Core layer causing the error in the issue

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a valid input (non-zero dimensions) to avoid the error mentioned in the issue
    return torch.rand(2, 3, 4)  # Example input with shape (batch=2, channels=3, length=4)

