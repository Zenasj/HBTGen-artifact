# torch.rand(1, dtype=torch.long)  # Sample size as a scalar tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters from the original test case
        self.total_count = torch.tensor(1.0)
        self.probs = torch.tensor(0.9)
        # Placeholder for comparison logic (original vs fixed)
        self.original_logic = nn.Identity()  # Stub for original algorithm
        self.fixed_logic = nn.Identity()     # Stub for fixed algorithm

    def forward(self, sample_size):
        # Simulate original and fixed versions (using current implementation for both as stub)
        # Actual implementation difference would be in C++ code, so using placeholder logic
        sample_shape = torch.Size([sample_size.item()])

        # Generate samples using current implementation (assumed to have the fix)
        current_dist = torch.distributions.Binomial(self.total_count, self.probs)
        current_samples = current_dist.sample(sample_shape)

        # Check for negative values as per the issue's bug report
        has_negative = (current_samples < 0).any()

        # Return boolean indicating presence of negative values (comparison result)
        return has_negative

def my_model_function():
    return MyModel()

def GetInput():
    # Returns sample size tensor matching the original test (1e9 samples)
    return torch.tensor([1000000000], dtype=torch.long)

