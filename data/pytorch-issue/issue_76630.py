import torch
input = torch.rand([5], dtype=torch.float32)
other = torch.rand([5, 5], dtype=torch.float64)

r1 = torch.maximum(input, other)
r2 = torch.clamp(input, min=other)
print(r1.dtype, r2.dtype)
# torch.float64 torch.float32