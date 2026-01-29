# torch.rand(1, dtype=torch.float32)  # Dummy input shape (assumed from benchmark context)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seed = 42  # Fixed seed to replicate "non-random" behavior
        self.old_rng = torch.Generator()  # RNG for old (non-random) generation
        
    def forward(self, x):
        # Replicate old "non-random" behavior: reset seed each call
        self.old_rng.manual_seed(self.seed)
        old_output = torch.randn(10000, generator=self.old_rng)
        
        # New "random" behavior: use default RNG without resetting
        new_output = torch.randn(10000)
        
        # Return boolean tensor indicating if outputs differ beyond tolerance
        return torch.tensor(
            not torch.allclose(old_output, new_output, atol=1e-5),
            dtype=torch.bool
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Return dummy input tensor matching expected input shape
    return torch.rand(1, dtype=torch.float32)

