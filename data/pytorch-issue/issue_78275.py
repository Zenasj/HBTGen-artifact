# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed S matrix from the issue example
        self.register_buffer('S', torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, z):
        B = z.shape[0]
        # Compute using three methods
        einsum_result = torch.einsum('bi,ij,bj->b', z, self.S, z)
        matmul_result = (z.view(B, 1, 2) @ self.S @ z.view(B, 2, 1)).squeeze(-1).squeeze(-1)
        manual_result = (
            z[:,0] * self.S[0,0] * z[:,0] +
            z[:,0] * self.S[0,1] * z[:,1] +
            z[:,1] * self.S[1,0] * z[:,0] +
            z[:,1] * self.S[1,1] * z[:,1]
        )
        
        # Check if results are close within tolerance (matches observed discrepancy ~0.002)
        tol = 1e-3
        all_close_einsum_matmul = torch.allclose(einsum_result, matmul_result, atol=tol)
        all_close_einsum_manual = torch.allclose(einsum_result, manual_result, atol=tol)
        return torch.tensor([all_close_einsum_matmul and all_close_einsum_manual], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate batched input with shape (B,2)
    B = 1  # Matches original issue's input size
    return torch.rand(B, 2, dtype=torch.float32)

