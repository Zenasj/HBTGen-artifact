import torch.nn as nn

3

import functools
import torch

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_mask,
    create_block_mask
)

# q, k, v all padded to the nearest multiple of 128
# q = tensor[1, 1, 11008, 16] n=176128 (0.7Mb) x∈[-4.484, 4.669] μ=-0.246 σ=1.532 grad ConstantPadNdBackward0 cuda:0
# k = tensor[1, 1, 40320, 16] n=645120 (2.5Mb) x∈[-4.777, 3.912] μ=-0.046 σ=1.436 grad ConstantPadNdBackward0 cuda:0
# v = tensor[1, 1, 40320, 16] n=645120 (2.5Mb) x∈[-5.212, 4.806] μ=0.361 σ=1.419 grad ConstantPadNdBackward0 cuda:0

mask = create_block_mask(..., None, None, q.shape[2], v.shape[2], _compile=True)

# the line above appears to compile correctly, and produces a BlockMask object:
# mask = BlockMask(
#     kv_num_blocks=torch.Size([1, 1, 86]),
#     kv_indices=torch.Size([1, 1, 86, 315]),
#     full_kv_num_blocks=torch.Size([1, 1, 86]),
#     full_kv_indices=torch.Size([1, 1, 86, 315]),
#     q_num_blocks=torch.Size([1, 1, 315]),
#     q_indices=torch.Size([1, 1, 315, 86]),
#     full_q_num_blocks=torch.Size([1, 1, 315]),
#     full_q_indices=torch.Size([1, 1, 315, 86]),
#     BLOCK_SIZE=(128, 128),
#     shape=(1, 1, 11008, 40320),
#     sparsity=98.59%,
#     mask_mod=bipartite_graph_mask
# )

def noop(score, b, h, q_idx, kv_idx):
    return score

attention_op = functools.partial(flex_attention, score_mod=noop, block_mask=mask) # cache the block mask
attention_op = torch.compile(attention_op, fullgraph=False)

att_out = attention_op(q, k, v, block_mask=mask).sum()
att_out.backward()

3
import os
import torch

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_mask,
    create_block_mask
)

import functools

# import lovely_tensors as lt
# lt.monkey_patch()

# torch.__version__ == '2.6.0.dev20241107+cu121'
device = "cuda"

q = torch.randn((1, 1, 11008, 16), dtype=torch.float32, device="cuda")
k = torch.randn((1, 1, 40320, 16), dtype=torch.float32, device="cuda")
v = torch.randn((1, 1, 40320, 16), dtype=torch.float32, device="cuda")

# just a random lookup table
lookup_table = torch.randint(0, 51608, (11008, 128), dtype=torch.int, device=device)

def graph_mask(b, h, q_idx, kv_idx):
    return torch.any(lookup_table[q_idx] == kv_idx)

mask = create_block_mask(graph_mask, None, None, q.shape[2], v.shape[2], _compile=True)

# BlockMask(
#   kv_num_blocks=torch.Size([1, 1, 86]),
#    kv_indices=torch.Size([1, 1, 86, 315]),
#   full_kv_num_blocks=torch.Size([1, 1, 86]),
#    full_kv_indices=torch.Size([1, 1, 86, 315]),
#    q_num_blocks=torch.Size([1, 1, 315]),
#    q_indices=torch.Size([1, 1, 315, 86]),
#    full_q_num_blocks=torch.Size([1, 1, 315]),
#    full_q_indices=torch.Size([1, 1, 315, 86]),
#    BLOCK_SIZE=(128, 128),
#    shape=(1, 1, 11008, 40320),
#    sparsity=0.00%,
#    mask_mod=graph_mask
# )

def noop(score, b, h, q_idx, kv_idx):
    return score

attention_op = functools.partial(flex_attention, score_mod=noop, block_mask=mask)
attention_op = torch.compile(attention_op, fullgraph=False)  # this line triggers the error