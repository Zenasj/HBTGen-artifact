# torch.rand(B, 12, 3, dtype=torch.double)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, atol=None, rtol=None):
        super().__init__()
        self.atol = atol
        self.rtol = rtol

    def slow_inverse_th(self, A):
        B, M, N = A.shape
        Ainv = torch.zeros((B, N, M), dtype=A.dtype, device=A.device)
        for i in range(B):
            Ainv[i] = torch.linalg.pinv(A[i], atol=self.atol, rtol=self.rtol)
        return Ainv

    def forward(self, A):
        slow_inv = self.slow_inverse_th(A)
        fast_inv = torch.linalg.pinv(A, atol=self.atol, rtol=self.rtol)
        diff = slow_inv - fast_inv
        max_diff = diff.abs().max()
        mean_diff = diff.abs().mean()
        return max_diff, mean_diff

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(30, 12, 3, dtype=torch.double)

