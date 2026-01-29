# torch.rand(B, 30, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Input shape after flattening: (B, 30*32*32) → in_features=30720
        self.fc = nn.Linear(30*32*32, 3)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (B, 30*32*32)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape described in the issue
    return torch.randn((20, 30, 32, 32)).to("cuda:0")

