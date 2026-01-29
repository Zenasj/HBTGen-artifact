# torch.rand(3,), torch.rand(1), torch.randint(0,4, (3,))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        sorted_sequence, _, sorter = inputs  # values unused for validity check
        # Check if all indices in 'sorter' are within valid range [0, len(sorted_sequence)-1]
        valid = torch.all((sorter >= 0) & (sorter < sorted_sequence.size(0)))
        return valid  # Returns True/False as tensor indicating agreement between methods

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a valid sorted sequence
    sorted_sequence = torch.sort(torch.rand(3))[0]
    # Generate random query values (scalar as tensor)
    values = torch.rand(1)
    # Create a 50% chance of invalid indices (>=3) in 'sorter'
    if torch.rand(1).item() > 0.5:
        # Invalid case: set last index to 3 (out of bounds for size 3)
        indices = torch.randint(0, 3, (3,))
        indices[-1] = 3
        sorter = indices
    else:
        # Valid case: random permutation of indices 0-2
        sorter = torch.randperm(3)
    return (sorted_sequence, values, sorter)

