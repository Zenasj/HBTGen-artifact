import torch.nn as nn

import torch
import torch._inductor.utils
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)


create_block_mask = torch.compile(create_block_mask)


def create_block_mask_from_seqlens(
    q_seqlen: torch.Tensor,
    kv_seqlen: torch.Tensor,
) -> BlockMask:

    device = q_seqlen.device

    B, H = None, None

    q_batch = torch.arange(q_seqlen.size(0), device=device).repeat_interleave(q_seqlen)
    kv_batch = torch.arange(kv_seqlen.size(0), device=device).repeat_interleave(
        kv_seqlen
    )

    Q_LEN = q_batch.size(0)

    KV_LEN = kv_batch.size(0)

    def batch_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ):
        q_idx_batch = q_batch[q_idx]
        kv_idx_batch = kv_batch[kv_idx]
        batch_mask = (
            (q_idx_batch == kv_idx_batch) & (q_idx_batch != -1) & (kv_idx_batch != -1)
        )

        return batch_mask

    return create_block_mask(
        batch_mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
    )


a = torch.tensor([2, 42, 18, 21, 4, 2, 7, 1, 1]).cuda()
b = torch.tensor([57, 21, 16, 8]).cuda()


with torch._inductor.utils.fresh_inductor_cache():
    for seqlen in [a, b]:
        create_block_mask_from_seqlens(q_seqlen=seqlen, kv_seqlen=seqlen)
        torch.cuda.synchronize()