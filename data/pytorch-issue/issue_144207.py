import torch.nn as nn

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

input_tensor = torch.randn(3, 5, 3)

# Note: The first sequence has a length smaller than its actual length (4 > 3)
lengths = [4, 2, 3]

packed = pack_padded_sequence(input_tensor, lengths, batch_first=True, enforce_sorted=False)
unpacked, unpacked_lengths = pad_packed_sequence(packed, batch_first=True)

# Outputs: (3, 4, 3)
print("Unpacked Sequence Shape:", unpacked.shape)       
# Outputs: [4, 2, 3]
print("Unpacked Lengths:", unpacked_lengths)             
print("Original Sequence:", input_tensor) 
# Note: the last sequence length inde has been truncated
print("Unpacked Sequence:", unpacked)