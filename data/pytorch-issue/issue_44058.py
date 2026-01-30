import torch

res = torch.zeros_like(input_coalesced, memory_format=torch.preserve_format)

res = torch.empty_like(input_coalesced, memory_format=torch.preserve_format)