import torch.nn as nn

import torch

device = torch.device("mps")
conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3).to(device)

# Errors
data = torch.rand(1, 176, 1, dtype=torch.float32)
x = data.permute(0, 2, 1).contiguous().to(device)

# Does not error
# x = torch.rand(1, 1, 176, dtype=torch.float32, device=device)

conv(x).sum().backward()