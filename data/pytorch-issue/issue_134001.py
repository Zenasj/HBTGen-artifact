# torch.rand(B, S, D, dtype=torch.float32)  # B=batch (2), S=sequence length (4), D=64
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = 64
        seqlen = 4
        self.d = d
        self.seqlen = seqlen
        self.q_proj = nn.Linear(d, d, bias=False)
        self.attnSDPA = AttentionSDPA(d=d, seqlen=seqlen)
    
    def forward(self, x):
        bs = x.size(0)
        x = self.q_proj(x).reshape([bs, self.seqlen, self.d // 64, 64])
        x = self.attnSDPA(x)
        return x

class AttentionSDPA(nn.Module):
    def __init__(self, d, seqlen):
        super().__init__()
        self.wo = nn.Linear(d, d)
        self.seqlen = seqlen
    
    def forward(self, x):
        x = x.transpose(1, 2)
        attended = F.scaled_dot_product_attention(x, x, x)
        attended = attended.transpose(1, 2).reshape([x.shape[0], self.seqlen, -1])
        return self.wo(attended)

def my_model_function():
    return MyModel()

def GetInput():
    bs = 2
    seqlen = 4
    d = 64
    return torch.rand(bs, seqlen, d, dtype=torch.float32, device="cuda")

