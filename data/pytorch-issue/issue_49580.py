# torch.rand(B, 1, 2, 2, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Compute Frobenius norm (old torch.norm behavior)
        norm_old = torch.norm(x, 2, dim=(2, 3))  # dims over last two dimensions (matrix)
        # Compute matrix 2-norm (torch.linalg.norm behavior)
        norm_new = torch.linalg.norm(x, ord=2, dim=(2, 3))
        # Return boolean tensor indicating if norms are close within tolerance
        return torch.isclose(norm_old, norm_new, atol=1e-5).squeeze()  # Squeeze for scalar output in single-element batches

def my_model_function():
    return MyModel()

def GetInput():
    # 4D input tensor matching (B, C, H, W) format with example-compatible dimensions
    return torch.rand(1, 1, 2, 2, dtype=torch.float)  # B=1, C=1 (grayscale), H=2, W=2

