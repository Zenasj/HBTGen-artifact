# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameter to trigger MPS memory allocation tracking
        self.param = nn.Parameter(torch.empty(3, 224, 224))  # Matches channel/shape of input
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)  # Example operation
    
    def forward(self, x):
        # Add parameter to input tensor for memory allocation demonstration
        x = x + self.param.unsqueeze(0)  # Expand to batch dimension
        return self.conv(x)

def my_model_function():
    # Initialize model with MPS compatibility if available
    model = MyModel()
    # Explicitly initialize parameters to avoid uninitialized memory
    for param in model.parameters():
        torch.nn.init.normal_(param)
    return model

def GetInput():
    B, C, H, W = 1, 3, 224, 224
    return torch.rand(B, C, H, W, dtype=torch.float32)

