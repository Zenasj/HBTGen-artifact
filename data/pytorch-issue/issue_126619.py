# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (1, 3, 5, 5)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('lower', torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32))
        self.register_buffer('upper', torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32))

    def forward(self, x):
        # Reshape and expand lower/upper bounds to (1, 3, 1, 1)
        lower_reshaped = self.lower.view(1, -1, 1, 1)
        upper_reshaped = self.upper.view(1, -1, 1, 1)
        
        # Element-wise comparisons
        ge = x >= lower_reshaped
        le = x <= upper_reshaped
        
        # Combine with logical AND
        mask = torch.logical_and(ge, le)
        
        # Check all channels meet condition, keep dimensions
        result = mask.all(dim=1, keepdim=True)
        
        # Convert to float32 as output
        return result.to(torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 5, 5, dtype=torch.float32)

