# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()
        self.fwd_count = 0

    def increment(self):
        self.fwd_count += 1

    def forward(self, x):
        try:
            return self.layer(x)
        finally:
            self.increment()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 4)  # Matches 4D input expected by the model

