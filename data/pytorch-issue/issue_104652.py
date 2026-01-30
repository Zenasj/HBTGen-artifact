import torch
import math

L = 10
S = 10
attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).bool()
attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask

Python

def ref_attention(
    query, key, value, is_causal, attn_mask=None, dropout_p=0.0, scale=None
):
    scale = 1 / math.sqrt(math.sqrt(query.size(-1))) if scale is None else scale
    query *= scale
    key *= scale
    attn_weights = torch.matmul(query, key.transpose(-2, -1))
    if is_causal:
        assert attn_mask is None
        temp_mask = (
            torch.ones((query.shape[-2], key.shape[-2]), device=query.device)
            .tril_()
            .bool()
        )
        mask = torch.zeros_like(temp_mask, dtype=query.dtype)
        mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_weights.add_(mask)

    if attn_mask is not None:
        attn_weights.add_(attn_mask)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p)
    return torch.matmul(attn_weights, value)