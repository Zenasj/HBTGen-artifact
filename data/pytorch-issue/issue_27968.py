import torch
import torch.nn as nn
import numpy as np

m = nn.MultiheadAttention(embed_dim=20, num_heads=5, dropout=0.1)

i = torch.randn(4, 5, 20)
q = k = v = i.transpose(0, 1)
key_padding_mask = seq_len_to_mask([5, 3, 1, 0], max_len=5)

ao, aow = m(q, k, v, key_padding_mask=key_padding_mask)
ao = ao.transpose(0, 1)

print(aow)

def seq_len_to_mask(seq_len, max_len=None, mask_pos_to_true=True):

    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask