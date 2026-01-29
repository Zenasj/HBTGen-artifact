# torch.rand(B, M, N, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute SVD with full_matrices=True
        u_full, s_full, vh_full = torch.linalg.svd(x, full_matrices=True)
        # Compute SVD with full_matrices=False
        u_part, s_part, vh_part = torch.linalg.svd(x, full_matrices=False)
        
        # Truncate full matrices to match the dimensions of partial outputs
        m, n = x.shape[-2], x.shape[-1]
        k = min(m, n)
        s_full_trunc = s_full[..., :k]
        u_full_trunc = u_full[..., :, :k]
        vh_full_trunc = vh_full[..., :k, :]
        
        # Compute differences between outputs of the two SVD variants
        s_diff = torch.norm(s_full_trunc - s_part)
        u_diff = torch.norm(u_part - u_full_trunc)
        vh_diff = torch.norm(vh_part - vh_full_trunc)
        total_diff = s_diff + u_diff + vh_diff
        
        return total_diff

def my_model_function():
    return MyModel()

def GetInput():
    # Using the first test case's shape (100,10,10) from the issue
    return torch.rand(100, 10, 10, dtype=torch.float)

