import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def generate_causal_offset(offset: torch.Tensor):

    def causal_offset_mask(b, h, q_idx, kv_idx):
        return (offset + q_idx) >= kv_idx

    return causal_offset_mask


prefill = 128
max_seq_new_tokens = 100
B, H, HEAD_DIM = 1, 16, 64
start_offset = torch.tensor(prefill, device="cuda", dtype=torch.int32)
query = torch.rand(B, 1, H, HEAD_DIM, device="cuda", dtype=torch.float16).transpose(
    1, 2
)
TRANSPOSE = True

for i in range(max_seq_new_tokens):
    if TRANSPOSE:
        key = torch.rand(
            B, prefill + i, H, HEAD_DIM, device="cuda", dtype=torch.float16
        ).transpose(1, 2)
        value = torch.rand(
            B, prefill + i, H, HEAD_DIM, device="cuda", dtype=torch.float16
        ).transpose(1, 2)
    else:
        key = torch.rand(B, H, prefill + i, HEAD_DIM, device="cuda", dtype=torch.float16)
        value = torch.rand(B, H, prefill + i, HEAD_DIM, device="cuda", dtype=torch.float16)

    # create a causal mask
    offset = start_offset + i
    causal_offset_mask = generate_causal_offset(offset)
    block_mask = create_block_mask(causal_offset_mask, 1, 1, 1, key.shape[-2])
    flex_compile = torch.compile(flex_attention)
    flex_eager = flex_attention

    out_compile = flex_compile(query, key, value, block_mask=block_mask)
    out_eager = flex_eager(query, key, value, block_mask=block_mask)
    print("out_compile contains nan", torch.isnan(out_compile).any())
    print("out_eager contains nan", torch.isnan(out_eager).any())
    try:
        torch.testing.assert_close(out_compile, out_eager, atol=1e-2, rtol=0.0)
    except AssertionError as e:
        print(f"Failed at {i}")
        raise e