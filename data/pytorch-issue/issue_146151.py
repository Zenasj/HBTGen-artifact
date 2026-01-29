# torch.rand(1, dtype=torch.float32).cuda()  # Input is a scalar v_mult value

import torch
from torch.nn.attention.flex_attention import flex_attention

class MyModel(torch.nn.Module):
    def forward(self, v_mult_tensor):
        v_mult = v_mult_tensor.item()
        bsz, n_head, seq_len, qk_dim = 4, 8, 256, 64
        v_dim = int(qk_dim * v_mult)
        query = torch.randn(bsz, n_head, seq_len, qk_dim, dtype=torch.bfloat16).cuda()
        key = torch.randn(bsz, n_head, seq_len, qk_dim, dtype=torch.bfloat16).cuda()
        value = torch.randn(bsz, n_head, seq_len, v_dim, dtype=torch.bfloat16).cuda()
        out = flex_attention(query, key, value)
        out = out.transpose(1, 2).reshape(bsz, seq_len, int(n_head * v_dim))
        return out

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor with v_mult=2.0 to trigger the error scenario
    return torch.tensor([2.0], dtype=torch.float32).cuda()

