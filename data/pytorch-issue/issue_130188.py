import torch.nn as nn

import torch
import torch.nn.functional as F

import torch
torch.set_default_device('cuda')

from functools import wraps

# Example usage
_DEFAULT_SPARSE_BLOCK_SIZE = 128

def broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x

def _create_mask(
    score_mod,
    B: int,
    H: int,
    M: int,
    N: int,
    device: str = "cuda",
    _compiled: bool = False,
):
    r"""This function creates a mask tensor from a score_mod function.

    Args:
        score_mod (Callable): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of heads.
        M (int): Sequence length of query.
        N (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.

    Returns:
        mask (Tensor): A mask tensor with shape (B, H, M, N).
    """
    from contextlib import nullcontext

    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)
    # TODO: fix this

    # A hack required because of lack of torchfunctionmode support
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None))

    out = score_mod(torch.zeros(B, H, M, N, device=device), b, h, m, n)
    mask = torch.where(torch.isneginf(out), False, True)
    return mask


def foo(qk, batch, head, q_idx, kv_idx):
    causal_mask = q_idx <= kv_idx
    return torch.where(causal_mask, qk * 1.234, -float("inf"))

mask = torch.compile(_create_mask)(foo, 4, 1, 1024, 1024, device='cuda')