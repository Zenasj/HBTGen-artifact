# torch.rand(1, 3, 2, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, stride=1, padding=1)
        # Ensure other_tensor is a buffer to handle device placement correctly
        self.register_buffer('other_tensor', torch.randn(2, 1, 1))

    def forward(self, x):
        v1 = self.conv(x)
        # Apply alpha=0.875 as specified in the issue
        v2 = torch.add(v1, self.other_tensor, alpha=0.875)
        return v2

def my_model_function():
    # Initialize and return the model instance
    return MyModel()

def GetInput():
    # Generate input matching the model's expected shape (B=1, C=3, H=2, W=2)
    return torch.randn(1, 3, 2, 2, dtype=torch.float32)

