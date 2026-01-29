# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple model to demonstrate safe nonzero usage
        self.threshold = 0.5  # Example parameter to generate a mask

    def forward(self, x):
        # Create a boolean mask where values exceed the threshold
        mask = x > self.threshold
        # Use nonzero with explicit as_tuple=False to avoid deprecation warning
        indices = torch.nonzero(mask, as_tuple=False)
        return indices

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random 4D tensor (B, C, H, W)
    return torch.rand(2, 3, 64, 64)  # Example shape (batch=2, channels=3, 64x64)

