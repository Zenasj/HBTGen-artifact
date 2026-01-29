# torch.rand(10, 3, 1000, 1000, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3)  # Matches original Conv2d parameters
        
    def forward(self, x):
        x = self.conv(x)
        # Apply unfold on dimension 3 (width) with window size 5 and step 1
        unfolded = x.unfold(3, 5, 1)
        return unfolded.sum()  # Matches original backward trigger

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 3, 1000, 1000, dtype=torch.float32)

