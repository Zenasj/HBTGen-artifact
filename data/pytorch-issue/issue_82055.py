import torch

c = torch.add(a,b)
# Works for both quantized and float tensors, scale and zero-point for c
# is set to 0 and 1.

# is set to 1 and 0.