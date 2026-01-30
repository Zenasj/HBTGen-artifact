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

def _convert_mask_to_block_mask(
    mask,
    KV_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE=_DEFAULT_SPARSE_BLOCK_SIZE,
):
    assert mask.dtype == torch.bool
    mask = broadcast_to_dim(mask, 4)
    B, H, Q, KV = mask.shape
    assert Q % Q_BLOCK_SIZE == 0
    assert KV % KV_BLOCK_SIZE == 0
    mask = mask.view(
        B, H, Q // Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV // KV_BLOCK_SIZE, KV_BLOCK_SIZE
    )  # [B, H, Q//Q_BLOCK_SIZE, Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.permute(
        0, 1, 2, 4, 3, 5
    )  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE, Q_BLOCK_SIZE, KV_BLOCK_SIZE]
    mask = mask.sum(dim=[-2, -1]) > 0  # [B, H, Q//Q_BLOCK_SIZE, KV//KV_BLOCK_SIZE]
    return mask

def _create_block_mask_from_mask(
    block_mask: torch.Tensor,
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
):
    device = block_mask.device
    block_mask = block_mask.to(dtype=torch.int8)
    kv_num_blocks = block_mask.sum(dim=3)
    kv_indices = torch.argsort(block_mask, dim=3, descending=True, stable=True)
    q_num_blocks = block_mask.sum(dim=2)
    q_indices = torch.argsort(block_mask, dim=2, descending=True, stable=True).permute(
        0, 1, 3, 2
    )
    return (
        kv_num_blocks.to(torch.int32).to(device).contiguous(),
        kv_indices.to(torch.int32).to(device).contiguous(),
        q_num_blocks.to(torch.int32).to(device).contiguous(),
        q_indices.to(torch.int32).to(device).contiguous(),
        KV_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    )

def _create_block_mask(
    score_mod,
    B: int,
    H: int,
    M: int,
    N: int,
    device: str = "cuda",
    KV_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
    Q_BLOCK_SIZE: int = _DEFAULT_SPARSE_BLOCK_SIZE,
):
    r"""This function creates a block mask tuple from a score_mod function.

    Args:
        score_mod (Callable): Function to modify attention scores.
        B (int): Batch size.
        H (int): Number of heads.
        M (int): Sequence length of query.
        N (int): Sequence length of key/value.
        device (str): Device to run the mask creation on.
        KV_BLOCK_SIZE (int): Block size of block mask for each query.
        Q_BLOCK_SIZE (int): Block size of block mask for each key/value.

    Returns:
        block_mask (tuple): A tuple of (kv_num_blocks, kv_indices, q_num_blocks, q_indices,
                            KV_BLOCK_SIZE, Q_BLOCK_SIZE) which represents the block mask.
    """
    mask = torch.where(torch.isneginf(score_mod(torch.zeros(1, 1, M, N), None, None, torch.arange(M).view(1, 1, M, 1), torch.arange(N).view(1, 1, 1, N))), False, True)
    # mask = _create_mask(score_mod, B, H, M, N, device)
    block_mask = _convert_mask_to_block_mask(
        mask, KV_BLOCK_SIZE=KV_BLOCK_SIZE, Q_BLOCK_SIZE=Q_BLOCK_SIZE
    )
    block_mask = _create_block_mask_from_mask(block_mask, KV_BLOCK_SIZE, Q_BLOCK_SIZE)
    return block_mask

document_id = torch.zeros(32768, dtype=torch.int, device='cuda')
document_id[:4096] = 0
for i in range(4096, 32768, 2048):
    document_id[i:i+2048] = i // 2048

def document_masking_causal(qk, batch, head, q_idx, kv_idx):
    causal_mask = q_idx <= kv_idx
    document_mask = (document_id[q_idx] == document_id[kv_idx])
    return torch.where(causal_mask & document_mask, qk, -float("inf"))

mask = torch.compile(_create_block_mask)(document_masking_causal, 4, 1, 32768, 32768, device='cuda')