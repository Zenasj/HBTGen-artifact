import torch
import torch.nn as nn
N = 2048
C = 384
H = 1
W = 1

groups = 3
input = torch.randn(H, N, C, W, requires_grad=True, device='cpu').permute(1, 2, 0, 3)
# input shape: torch.Size([2048, 384, 1, 1]), stride: (384, 1, 786432, 1)
print(f"input shape: {input.shape}, stride: {input.stride()}")
# True
print(input.is_contiguous(memory_format=torch.channels_last))

m = nn.GroupNorm(groups, C, affine=False, device='cpu')
output = m(input)
gradient = torch.randn(N, C, H, W, device='cpu').to(memory_format=torch.channels_last)
# gradient shape: torch.Size([2048, 384, 1, 1]), stride: (384, 1, 384, 384)
print(f"gradient shape: {gradient.shape}, stride: {gradient.stride()}")
output.backward(gradient) # RuntimeError: Expected memory formats of X and dY are same.