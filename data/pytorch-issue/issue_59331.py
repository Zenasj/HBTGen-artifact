# torch.rand(512, 8, 64, dtype=torch.float16, device='cuda'), torch.rand(512, 64, 8, dtype=torch.float16, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        mat_a, mat_b = inputs
        # Loop-based approach (equivalent to original code's for-loop)
        res_loop = torch.stack([torch.mm(a, b) for a, b in zip(mat_a, mat_b)])
        # Batched approach using torch.bmm
        res_bmm = torch.bmm(mat_a, mat_b)
        # Calculate maximum absolute difference across all elements
        max_diff = (res_bmm - res_loop).abs().max()
        # Return whether difference is within 0.01 tolerance (as observed in comments)
        return max_diff < 0.01  # Returns boolean tensor (scalar)

def my_model_function():
    return MyModel()

def GetInput():
    mat_a = torch.rand(512, 8, 64, dtype=torch.float16, device='cuda')
    mat_b = torch.rand(512, 64, 8, dtype=torch.float16, device='cuda')
    return (mat_a, mat_b)

