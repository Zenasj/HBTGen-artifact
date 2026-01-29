# torch.rand(B, C, dtype=torch.bfloat16, device='cuda', requires_grad=True)
import torch
from torch import nn
from xformers.checkpoint import selective_checkpoint_wrapper

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weight parameter matching the example's w.t() shape (2048x2048)
        self.w = nn.Parameter(torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda"))

    def forward(self, t):
        b, _ = t.shape
        idxs = torch.randperm(b, device=t.device)[:b // 2]
        sub_in = t[idxs]
        sub_out = torch.matmul(sub_in, self.w)
        out = t.index_add(0, idxs, sub_out, alpha=0.5)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((512, 2048), dtype=torch.bfloat16, device='cuda', requires_grad=True)

