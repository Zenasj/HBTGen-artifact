# torch.rand(1, 128, 512, dtype=torch.bfloat16, device='cuda', requires_grad=True)
import torch
import torch.nn.functional as F
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 8
        self.n_kv_heads = 8
        self.dim = 512
        self.head_dim = self.dim // self.n_heads  # 64
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bs, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.n_kv_heads, self.head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)

def my_model_function():
    model = MyModel()
    model.to('cuda')  # Explicitly move to CUDA as in original setup
    return model

def GetInput():
    return torch.randn(1, 128, 512, dtype=torch.bfloat16, device='cuda', requires_grad=True)

