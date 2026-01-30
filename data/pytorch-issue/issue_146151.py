import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention

class Model(torch.nn.Module):
    def forward(self, v_mult=1):
        bsz, n_head, seq_len, qk_dim = 4, 8, 256, 64
        v_dim = int(qk_dim * v_mult)
        query = torch.randn(bsz, n_head, seq_len, qk_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(bsz, n_head, seq_len, qk_dim, dtype=torch.bfloat16).cuda()
        value = torch.randn(bsz, n_head, seq_len, v_dim, dtype=torch.bfloat16).cuda()
        
        out = flex_attention(query, key, value)
        out = out.transpose(1, 2).reshape(bsz, seq_len, int(n_head * v_dim)) # [bsz, num_heads, slen, v_head_dim] -> [bsz, slen, num_heads * v_head_dim]
        return out.shape

mod = Model().cuda()
mc = torch.compile(mod)

for v_mult in [1, 0.5, 2]:
    print(f"v_mult = {v_mult}")
    
    print(mod)
    print(mod(v_mult))
    print(mc(v_mult))