# torch.rand(1, 1024, 768, dtype=torch.float16)  # Input shape: B=1, S=1024, D=768 (12 heads * 64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        D_MODEL = 12 * 64  # 768
        self.attn_qkvw = nn.Parameter(torch.normal(0, 1, size=(D_MODEL, D_MODEL * 3), device='cuda', dtype=torch.float16))
        self.attn_qkvb = nn.Parameter(torch.normal(0, 1, size=(D_MODEL * 3,), device='cuda', dtype=torch.float16))
        
    def forward(self, hidden_states):
        qkv = torch.matmul(hidden_states, self.attn_qkvw) + self.attn_qkvb
        assert qkv.is_contiguous()
        return qkv

def my_model_function():
    return MyModel()

def GetInput():
    return torch.normal(0, 1, size=(1, 1024, 12*64), device='cuda', dtype=torch.float16)

