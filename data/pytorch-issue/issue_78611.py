import torch

# Code to reproduce:
input = torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
input = input.reshape([1, 2, 3, 4])