import torch

if attn_mask is not None:
    if attn_mask.dtype == torch.bool:
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    else:
        attn_bias += attn_mask