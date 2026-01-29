# torch.rand(B, m, n, dtype=torch.float32)  # e.g., (3, 4, 5)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_tensor):
        Q, R = torch.qr(input_tensor, some=False)
        Q_T = Q.transpose(-2, -1)
        Q_T_Q = torch.matmul(Q_T, Q)
        n = Q.size(-1)
        batch_shape = Q.shape[:-2]
        identity = torch.eye(n, dtype=Q.dtype, device=Q.device).expand(batch_shape + (n, n))
        is_close = torch.allclose(Q_T_Q, identity, atol=1e-5)
        return torch.tensor(is_close, dtype=torch.bool).unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4, 5)

