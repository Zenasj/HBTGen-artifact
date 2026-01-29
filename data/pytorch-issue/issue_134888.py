# torch.rand(1, 512, 256, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(256, 256 * 3)
        self.n_head = 256 // 64
        self.d_attn = 256

    def forward(self, x):
        n_batch, n_ctx, _ = x.shape
        q, k, v = self.qkv_proj(x).split([self.d_attn, self.d_attn, self.d_attn], dim=2)
        q = q.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        k = k.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        v = v.reshape(n_batch, n_ctx, self.n_head, -1).transpose(1, 2)
        return nn.functional.scaled_dot_product_attention(q, k, v)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 512, 256), requires_grad=True)

