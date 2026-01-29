# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple model to demonstrate valid TorchScript usage
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Process input tensor (avoids problematic list-to-tuple conversions)
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 4D tensor compatible with the model's expected input
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

