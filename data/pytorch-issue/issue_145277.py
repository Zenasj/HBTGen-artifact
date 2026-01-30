import torch.nn as nn

import torch
from torch import nn

class LightningAttention(nn.Module):
    def forward(self, query, key=None, value=None, mask=None):
        batch_size, seq_len, dim = query.shape
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (dim ** 0.5)
        if mask != None: # switch to is not and suddenly it works fine
            scores = scores + mask
            
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, value)

batch_size, seq_len, dim = 2, 4, 8
x = torch.randn(batch_size, seq_len, dim)

model = LightningAttention()
try:
    model = torch.compile(
        model,
        backend='inductor',
        dynamic=False,
        fullgraph=True,
        options={
            "epilogue_fusion": True,
            "max_autotune": True,
        }
    )
    mask = torch.zeros(batch_size, seq_len, seq_len)
    output = model(x, x, x, mask)
except Exception as e:
    print(f"Error: {e}")