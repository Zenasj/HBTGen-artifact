import torch
import torch.nn as nn

# torch.rand(1, 3, 32, 32, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat([x, x])  # Concatenation of same node causing recursion error during quantization
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

