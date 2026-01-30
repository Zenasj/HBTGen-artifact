import torch.nn as nn

import random

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.nn.parallel.scatter_gather import _is_namedtuple

def _ordered_sequence(tensor_type):
    seqs = [tensor_type(random.randint(1, 256))
            for _ in range(32)]
    seqs = [s.random_(-128, 128) for s in seqs]
    ordered = sorted(seqs, key=len, reverse=True)
    return ordered

def _padded_sequence(tensor_type):
    ordered = _ordered_sequence(tensor_type)
    lengths = [len(i) for i in ordered]
    padded_tensor = rnn_utils.pad_sequence(ordered)
    return padded_tensor, lengths

padded, lengths = _padded_sequence(torch.Tensor)
packed = rnn_utils.pack_padded_sequence(
    padded, lengths, enforce_sorted=False)
print(type(packed), packed.data.device)
print(_is_namedtuple(packed))