import torch
import einops
import functools
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)


def _flex_attention_mask(b, h, q_idx, kv_idx, input_lengths):
    padding_condition = (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])
    return padding_condition

class Model(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()

        self.subsampler = torch.nn.Conv1d(256, 256, 5)
        self.projector = nn.Linear(256, dim)
        self.num_heads = 4

    def forward(self, x, input_lengths):
        x = self.subsampler(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.projector(x).transpose(0, 1)
        x = einops.rearrange(x, "time batch (heads d) -> batch heads time d", heads=self.num_heads)

        max_time = x.size(-2)
        mask = torch.compile(create_block_mask, dynamic=True)(
            functools.partial(
                _flex_attention_mask,
                input_lengths=input_lengths,
            ),
            B=torch.as_tensor(len(input_lengths), device=input_lengths.device),
            H=None,  # invariant
            Q_LEN=torch.as_tensor(max_time, device=input_lengths.device),
            KV_LEN=torch.as_tensor(max_time, device=input_lengths.device),
        )

        x = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            query=x, key=x, value=x, block_mask=mask
        )

        return x

model = Model(128).cuda()
B = 16
F = 256
T = 12
x = torch.randn(B, T, F, device='cuda')
l = torch.randint(0, T, (B, ), device='cuda')
model(x, l).shape

x = torch.compile(flex_attention, dynamic=True, fullgraph=True)(
            query=x, key=x, value=x, block_mask=mask
        )