import random

import torch
import os
import einops
import functools
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    create_mask,
    _round_up_to_multiple as round_up_to_multiple
)

torch.random.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision('highest')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

class MHA(nn.Module):
    def __init__(self, hid=64, nheads=4, use_flex=False):
        super().__init__()
        self.proj = nn.Linear(
            hid,
            3 * hid
        )
        self.out_proj = nn.Linear(
            hid,
            hid
        )

        self.hid = hid
        self.nheads = nheads
        self.head_dim = hid // nheads
        assert hid % nheads == 0

        self.use_flex = use_flex
        self.scaling = 1 / (self.head_dim ** 0.5)

    def forward(self, x, attn_mask):
        q, k, v = self.proj(x).chunk(3, -1)
        pattern = "b t (n h) -> b n t h"
        q = einops.rearrange(
            x,
            pattern,
            n=self.nheads,
            h=self.head_dim,
        )
        k = einops.rearrange(
            x,
            pattern,
            n=self.nheads,
            h=self.head_dim,
        )
        v = einops.rearrange(
            x,
            pattern,
            n=self.nheads,
            h=self.head_dim,
        )

        if self.use_flex:
            x = flex_attention(q, k, v, block_mask=attn_mask, scale=self.scaling,)
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask.unsqueeze(1),
                scale=self.scaling,
            )
        x = einops.rearrange(
            x,
            "b n t h -> t b (n h)",
            n=self.nheads,
            h=self.head_dim,
        )
        return self.out_proj(x)


def _flex_attention_chunked_mask_generator(b, h, q_idx, kv_idx, block_size, left_context_blocks_count, input_lengths):
    q_block_idxes = torch.div(q_idx, block_size, rounding_mode="floor")
    kv_block_idxes = torch.div(kv_idx, block_size, rounding_mode="floor")
    diff = q_block_idxes - kv_block_idxes

    blocks = (diff >= 0) & (diff < left_context_blocks_count)
    if input_lengths is None:
        return blocks

    padding_condition = (q_idx < input_lengths[b]) & (kv_idx < input_lengths[b])
    return blocks & padding_condition


def lengths_to_flex_attn_mask(
    input_lengths: torch.Tensor,
    block_context: tuple[int, int],
    device,
    max_time: int | None = None,
):
    left_context, right_context = block_context
    block_size = (right_context + 1)
    left_context = left_context
    assert left_context % block_size == 0
    left_context_blocks_count = left_context // block_size

    if max_time is None:
        assert input_lengths is not None
        max_time = torch.max(input_lengths).item()

    sparse_block_size = block_size
    sparse_block_size = round_up_to_multiple(sparse_block_size, 128)
    max_time = round_up_to_multiple(max_time, sparse_block_size)

    block_mask = create_block_mask(
        functools.partial(
            _flex_attention_chunked_mask_generator,
            block_size=torch.as_tensor(block_size, device=device),
            left_context_blocks_count=torch.as_tensor(left_context_blocks_count, device=device),
            input_lengths=torch.as_tensor(input_lengths, device=device) if input_lengths is not None else None,
        ),
        device=device,
        B=len(input_lengths) if input_lengths is not None else None,
        H=None,  # invariant
        Q_LEN=max_time,
        KV_LEN=max_time,
        BLOCK_SIZE=sparse_block_size,  # this is crucial to have full blocks
    )
    return block_mask


def lengths_to_attn_mask(
    input_lengths: torch.Tensor,
    block_context: tuple[int, int],
    device,
    max_time: int | None = None,
):
    if device is None:
        device = input_lengths.device

    if max_time is None:
        max_time = torch.max(input_lengths).item()

    left_context, right_context = block_context
    block_size = (right_context + 1)
    left_context = left_context
    assert left_context % block_size == 0
    left_context_blocks_count = left_context // block_size

    block_idxes = torch.div(torch.arange(max_time), block_size, rounding_mode="floor")
    block_idxes_diff = block_idxes.unsqueeze(1) - block_idxes.unsqueeze(0)
    attn_mask = (block_idxes_diff >= 0) & (block_idxes_diff < left_context_blocks_count)
    attn_mask = ~attn_mask
    key_padding_mask = (torch.arange(T).unsqueeze(0) >= input_lengths.unsqueeze(-1)).unsqueeze(-1)
    return ~(attn_mask.unsqueeze(0) | (key_padding_mask | key_padding_mask.transpose(-1, -2)))

heads = 8
B, T, features = 32, 64, 64 * heads

x = torch.randn(B, T, features)
input_lengths = torch.tensor([64, 47, 35, 31, 36,  3, 35, 18,  5, 49, 51, 16, 31, 58, 16, 11, 21, 10,
        44, 61,  1,  6, 34, 36, 40, 41, 59,  2, 58, 21,  5,  4])
max_time = x.size(1)
block_context = (90, 9)

flex_mask = lengths_to_flex_attn_mask(
    input_lengths=input_lengths,
    block_context=block_context,
    device='cuda',
    max_time=max_time,
)
mask_mod = flex_mask.mask_mod
flex_mask_dense = create_mask(mask_mod, B=B, H=None, Q_LEN=max_time, KV_LEN=max_time, device='cuda').squeeze(1).long().cpu()

regular_mask = lengths_to_attn_mask(
    input_lengths=input_lengths,
    block_context=block_context,
    device='cpu',
    max_time=max_time,
)

torch.testing.assert_close(
    regular_mask.long(),
    flex_mask_dense
)
default_model = MHA(
    hid=features,
    nheads=heads,
    use_flex=False
).eval().cpu()

flex_model = MHA(
    hid=features,
    nheads=heads,
    use_flex=True
).eval().cuda()
flex_model.load_state_dict(default_model.state_dict())

torch.testing.assert_close(
    actual=default_model(x, attn_mask=regular_mask).cuda(),
    expected=flex_model(x.cuda(), attn_mask=flex_mask)
)

torch.testing.assert_close(
    actual=default_model(x, attn_mask=regular_mask).cuda(),
    expected=torch.compile(flex_model)(x.cuda(), attn_mask=flex_mask)
)