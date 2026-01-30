import torch.nn as nn

import torch
from torch import nn
import torch._dynamo.config

torch._dynamo.config.verbose = True


class MyModule(nn.Module):

    def __init__(self,) -> None:
        super().__init__()
        self.self_attn_t = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.self_attn_f = nn.MultiheadAttention(128, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor):
        B, F, T, H = x.shape
        x = x.reshape(B * F, T, H)
        x, _ = self.self_attn_t(x, x, x)
        x = x.reshape(B, F, T, H)
        x = x.permute(0, 2, 1, 3)  # [B,T,F,H]
        x = x.reshape(B * T, F, H)
        x, _ = self.self_attn_f(x, x, x)
        x = x.reshape(B, T, F, H)
        x = x.permute(0, 2, 1, 3) # [B,F,T,H]
        return x


B, F, T, H = 2, 129, 100, 128
x = torch.randn((B, F, T, H)).cuda()
m = MyModule().cuda()
m = torch.compile(m, dynamic=True)
y = m(x)