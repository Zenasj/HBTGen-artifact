# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, split_size=2):
        super().__init__()
        self.split_size = split_size

    def forward(self, x):
        splits = torch.split(x, self.split_size, dim=1)  # Split along channels
        B = x.size(0)
        mask = torch.rand(B, self.split_size, device=x.device) < 0.5  # Example mask
        
        # Process first split to demonstrate dtype-aware masking
        first_split = splits[0]
        reduced = torch.sum(first_split, dim=(2, 3))  # Sum spatial dimensions
        
        # Apply mask with explicit dtype matching
        default_val = torch.tensor(0, dtype=reduced.dtype, device=reduced.device)
        masked_result = torch.where(
            mask[:, :reduced.size(1)], 
            reduced, 
            default_val
        )
        
        return masked_result

def my_model_function():
    return MyModel(split_size=2)

def GetInput():
    # Input shape matching the model's expected input (B, C, H, W)
    return torch.rand(2, 3, 4, 4, dtype=torch.float32)

