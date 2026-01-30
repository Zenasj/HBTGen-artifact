import torch.nn as nn

import torch
from torch.nn.utils.rnn import pack_padded_sequence
seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]], device='cuda')
lens = torch.tensor([2, 1, 3], device='cuda')
pack_padded_sequence(seq, lens, enforce_sorted=False)
# RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor

seq = torch.tensor([[1,2,0], [3,0,0], [4,5,6]], device='cuda')
lens = torch.tensor([2, 1, 3], device='cuda')
pack_padded_sequence(seq, lens, enforce_sorted=False)
# PackedSequence(data=tensor([0, 1, 2, 0, 3, 6], device='cuda:0'), batch_sizes=tensor([3, 2, 1]), sorted_indices=tensor([2, 0, 1], device='cuda:0'), unsorted_indices=tensor([1, 2, 0], device='cuda:0'))