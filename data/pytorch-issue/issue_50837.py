# torch.rand(B, N, N, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.linalg.cholesky  # Example op requiring SPD input

    def forward(self, x):
        # Process input to ensure it's symmetric positive definite (SPD)
        processed_x = x @ x.mT  # Symmetrize
        processed_x = processed_x + torch.eye(x.size(-1), dtype=x.dtype, device=x.device)  # Ensure positive definite
        return self.op(processed_x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input shape: batch_size=2, matrix size 3x3
    B, N = 2, 3
    return torch.rand(B, N, N, dtype=torch.float32)

