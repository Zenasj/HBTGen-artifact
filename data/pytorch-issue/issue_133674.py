import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Input is a tuple of three tensors with shapes (1, 8, 256, 128), (1, 8, 256, 128), (1, 8, 256, 512), dtype=torch.bfloat16, device="cuda"
class MyModel(nn.Module):
    def __init__(self, B, H, S, D, D_L):
        super().__init__()
        self.B = B
        self.H = H
        self.S = S
        self.D = D
        self.D_L = D_L
        # Define mask function matching the issue's mask_fn
        self.mask_fn = lambda b, h, q_idx, kv_idx: q_idx == kv_idx
        # Precompute block_mask using model parameters
        self.block_mask = create_block_mask(
            self.mask_fn,
            B=B,
            H=H,
            Q_LEN=S,
            KV_LEN=S
        )

    def forward(self, inputs):
        query, key, value = inputs
        return flex_attention(query, key, value, block_mask=self.block_mask)

def my_model_function():
    return MyModel(B=1, H=8, S=256, D=128, D_L=512)

def GetInput():
    B, H, S, D, D_L = 1, 8, 256, 128, 512
    query = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    key = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    value = torch.randn(B, H, S, D_L, device="cuda", dtype=torch.bfloat16, requires_grad=False)
    return (query, key, value)

