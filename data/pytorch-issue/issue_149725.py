import torch.nn as nn

py
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.o = torch.nn.Linear(64, 128)

    def forward(self, q, k, v, mask):
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
            out = scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=mask,
                is_causal=False,
                scale=1.0,
            )

        return out


model = Model().to("cuda:0")
model = torch.compile(model)
q = torch.randn(32, 1, 10, 64).to("cuda:0")
k = torch.randn(32, 1, 6, 64).to("cuda:0")
v = torch.randn(32, 1, 6, 64).to("cuda:0")
mask = torch.ones(32, 1, 10, 6).to("cuda:0")

model(q, k, v, mask)