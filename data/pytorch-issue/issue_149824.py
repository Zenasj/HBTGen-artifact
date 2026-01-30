import torch

print(q.shape)

torch.Size([64, 4, 64, 16])     # (h*w//window_size**2, heads, window_size**2, dim//heads