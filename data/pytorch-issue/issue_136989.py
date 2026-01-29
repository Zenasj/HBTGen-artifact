import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# torch.rand(B, C, H, W, dtype=...) → Input shape: (4, 1, 2048, 64)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        B = 4
        H = 1
        Q_LEN = 2048
        KV_LEN = 2048
        self.attn_mask = torch.ones((B, H, Q_LEN, KV_LEN), dtype=torch.bool, device='cuda').tril()

        def causal(b, h, q_idx, kv_idx):
            # Use the precomputed mask with correct device and index
            return self.attn_mask[b][0][q_idx][kv_idx]  # H is fixed to 1, so h is 0

        self.block_mask = create_block_mask(causal, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN)
        # Pre-initialize k and v as parameters to avoid dynamic input splitting
        self.k = nn.Parameter(torch.randn(B, H, Q_LEN, 64, device='cuda'))
        self.v = nn.Parameter(torch.randn(B, H, Q_LEN, 64, device='cuda'))

    def forward(self, q):
        return flex_attention(q, self.k, self.v, block_mask=self.block_mask)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor matching the input shape (B, H, L, E) = (4, 1, 2048, 64)
    return torch.rand(4, 1, 2048, 64, dtype=torch.float32, device='cuda')

