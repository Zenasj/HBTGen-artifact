import torch.nn as nn

import torch

gpu_device = torch.device("cuda:0")
batch_input = torch.zeros([3, 4]).to(gpu_device) # [batch_size, seq_length]
seq_length = torch.tensor([2, 1, 4])  # CPU by default

# Works
torch.nn.utils.rnn.pack_padded_sequence(batch_input, seq_length, batch_first=True, enforce_sorted=False)

# Fails
torch.nn.utils.rnn.pack_padded_sequence(batch_input, seq_length.to(gpu_device), batch_first=True, enforce_sorted=False)