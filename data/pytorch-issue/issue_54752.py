# torch.rand(1, 1)  # Input shape is arbitrary since the model is a pass-through
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Reproduce the segfault condition from the issue
        torch.set_num_threads(4)  # Critical value causing the bug
        # Create DataLoader that triggers the problem during initialization
        self.dataloader = torch.utils.data.DataLoader(
            [1, 2, 3],  # Dummy dataset
            num_workers=1  # Non-zero workers required for the bug
        )
        
    def forward(self, x):
        # Dummy forward pass (model doesn't process input)
        return x

def my_model_function():
    # Returns model instance with problematic configuration
    return MyModel()

def GetInput():
    # Returns a minimal valid input (model doesn't use it)
    return torch.rand(1, 1)

