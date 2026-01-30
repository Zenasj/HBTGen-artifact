import torch

input = torch.rand([1, 1], dtype=torch.complex32)
input.storage()
# RuntimeError: unsupported Storage type

input = torch.rand([1, 1], dtype=torch.complex32)
input.storage_offset()
# 0