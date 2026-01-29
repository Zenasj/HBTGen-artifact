# torch.randint(0, 2, (2, 2, 2, 6), dtype=torch.long)  # Inferred input shape (B=2, C=2, H=2, W=6)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute PyTorch's unique along dim=-1
        pt_unique = torch.unique(x, dim=-1)
        
        # Compute numpy-like unique by reshaping to 2D, applying unique, then reshaping back
        orig_shape = x.shape
        x_reshaped = x.view(-1, orig_shape[-1])  # Flatten all except last dim
        np_like_2d = torch.unique(x_reshaped, dim=1)  # Apply unique on 2D
        
        # Reshape back to original dimensions except the last (dim=-1) is now new size
        new_shape = list(orig_shape)
        new_shape[-1] = np_like_2d.size(1)
        np_like = np_like_2d.view(new_shape)
        
        # Compare outputs directly (without sorting) to detect order discrepancies
        return torch.all(pt_unique == np_like)

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a random tensor matching the input shape (2, 2, 2, 6) with 0/1 values
    return torch.randint(0, 2, (2, 2, 2, 6), dtype=torch.long)

