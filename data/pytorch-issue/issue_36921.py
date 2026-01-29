# torch.rand(B, 1109, 1109, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, A):
        n = A.size(1)
        B = torch.eye(n, dtype=A.dtype, device=A.device).expand(A.shape[0], n, n)
        LU, pivots = torch.lu(A)
        return torch.lu_solve(B, LU, pivots)

def my_model_function():
    return MyModel()

def GetInput():
    B_size = 1
    n = 1109
    A = torch.rand(B_size, n, n, dtype=torch.float32).cuda()
    A += 10 * torch.eye(n, dtype=torch.float32, device=A.device).unsqueeze(0)
    return A

