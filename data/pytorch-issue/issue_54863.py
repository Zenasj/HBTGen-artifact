# torch.randint(0, 2, (3, 4), dtype=torch.bool)  # Input is a boolean tensor of shape (3,4)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed indices for demonstration (rows [0,2], column 1)
        self.indices = (torch.tensor([0, 2], dtype=torch.long), torch.tensor([1], dtype=torch.long))

    def forward(self, mask):
        # Update selected indices to True (scalar boolean)
        updates = torch.tensor(True, dtype=torch.bool)
        return mask.index_put(self.indices, updates, accumulate=False)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random boolean mask tensor of shape (3,4)
    return torch.randint(0, 2, (3, 4), dtype=torch.bool)

