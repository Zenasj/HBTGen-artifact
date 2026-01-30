import torch.nn as nn

import torch

device = "mps"
m = torch.nn.MaxPool1d(3, stride=2).to(device)
input = torch.randn(20, 16, 50, device=device)


print("Shape: ", input.shape)
print("Shape: ", m(input).shape)
print("Shape: ", m(input.transpose(1,2)).shape)

print("Shape: ", m(input.transpose(1,2).clone(memory_format=torch.contiguous_format)).shape)