import torch.nn as nn

import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

seqs = [torch.randn(i, 1) for i in range(10, 0, -1)]
seqs = pack_sequence(seqs, enforce_sorted=False)
seqs = seqs.to('cuda') # .to('cuda') causes the bug, but .to(device='cuda') won't
print(seqs.sorted_indices.device)

try:
    pad_packed_sequence(seqs, True, 0)
except Exception as e:
    print(e)

import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

seqs = [torch.randn(i, 1) for i in range(10, 0, -1)]
seqs = pack_sequence(seqs, enforce_sorted=False)
seqs = seqs.to('cuda')
print(seqs.sorted_indices.device)

try:
    pad_packed_sequence(seqs, True, 0)
except Exception as e:
    print(e)