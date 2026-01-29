# torch.rand(B, N, N, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        try:
            # Attempt Cholesky decomposition; returns L if successful
            return torch.linalg.cholesky(x)
        except RuntimeError:
            # Return NaN-filled tensor to indicate failure (compatible with torch.compile)
            return torch.full_like(x, float('nan'))

def my_model_function():
    # Returns a model instance that performs Cholesky decomposition with failure handling
    return MyModel()

def GetInput():
    # Generates a batched square matrix with 50% chance of being non-PD
    B, N = 1, 100  # Batch size and matrix dimension
    a = torch.rand(B, N, N, dtype=torch.float32)
    H = a @ a.transpose(-2, -1)  # Ensure symmetric
    if torch.rand(1) > 0.5:
        # 50% chance to make non-PD by subtracting negative diagonal
        H -= 10 * torch.eye(N, dtype=torch.float32).unsqueeze(0)
    return H

