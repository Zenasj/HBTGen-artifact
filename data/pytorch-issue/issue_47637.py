import torch
import torch.nn as nn

sequences = torch.randn(10, 64, 128, device='cuda')
lengths = torch.randint(5, 10, [64], device='cuda')
torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths, enforce_sorted=False)