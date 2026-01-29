# torch.rand(2, 3, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Problematic approach (in-place PyTorch)
        x_pt = x.clone()
        x_pt -= x_pt[:, 0].unsqueeze(1)
        
        # Correct approach (non-in-place, mimics NumPy behavior)
        x_np = x - x[:, 0].unsqueeze(1)
        
        # Return comparison result as a boolean tensor
        return torch.tensor([torch.allclose(x_pt, x_np, atol=1e-5)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float)

