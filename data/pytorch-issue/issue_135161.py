# torch.rand(1, 512, 256, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
from torch.nn.attention.flex_attention import create_block_mask

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256
        self.qkv_proj.weight.data.fill_(0.1)
        self.qkv_proj.bias.data.fill_(0.1)
        self.mask1 = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx, 1, None, 512, 512, _compile=False
        )
        self.mask2 = create_block_mask(
            lambda b, h, q_idx, kv_idx: (q_idx >= kv_idx) & (q_idx <= kv_idx + 32), 1, None, 512, 512, _compile=False
        )

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)

        output1 = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask1)
        output2 = torch.nn.attention.flex_attention.flex_attention(q, k, v, block_mask=self.mask2)
        return output1 + output2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 512, 256), requires_grad=True)

