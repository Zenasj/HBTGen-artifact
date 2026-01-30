import torch.nn as nn

import torch

class Attention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d_model, num_heads)
    
    def forward(self, q, k, v):
        return self.attn(q, k, v)


model = Attention(1024, 16).cuda()
model_opt = torch.compile(model, dynamic=True)
torch._dynamo.config.verbose=True
x = torch.randn(1, 1, 1024).cuda()
model_opt(x, x, x)