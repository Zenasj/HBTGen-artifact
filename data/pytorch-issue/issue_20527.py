# torch.rand(B, 3, H, W, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)  # Matches the model structure in the issue's example
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor with 3 channels (as per Conv2d input requirements)
    return torch.rand(2, 3, 5, 5, dtype=torch.float)  # Example input shape (B=2, C=3, H=5, W=5)

