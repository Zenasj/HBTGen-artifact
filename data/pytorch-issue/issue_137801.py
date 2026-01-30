import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# note: 2046 % 128 (default block size) != 0
B, H, S, D = 8, 8, 2046, 64

query = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
key = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)
value = torch.randn(B, H, S, D, device="cuda", dtype=torch.float32)

random_mask = torch.randint(0, 2, size=(S,), device="cuda", dtype=torch.bool)

def random_mask_mod(b, h, q_idx, kv_idx):
    # mask based on q_idx. There are S entries in the random_mask lookup and the
    # expectation is that q_idx will be provided in the [0, S) range.
    return random_mask[q_idx]

# errors with:
# .../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [3,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
# .../aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [3,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
block_mask = create_block_mask(random_mask_mod, 1, 1, S, S, device=query.device)