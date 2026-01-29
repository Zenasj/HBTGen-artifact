# torch.randint(0, 2, (2,), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the buffer with the same shape as in the original example
        self.register_buffer('stored_tensor', torch.rand(2, 2, 3))

    def forward(self, t2):
        mask = t2 < 1  # Create boolean mask from input
        # Apply the mask to the stored tensor using __getitem__ (problematic operation)
        return self.stored_tensor[mask]

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching the original example's t2
    return torch.randint(0, 2, (2,), dtype=torch.int64)

