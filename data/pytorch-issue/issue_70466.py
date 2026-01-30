import torch

img = torch.rand((3, 32, 32))
# Pick any different channel order
t_img = img[[2, 1, 0]]
print((img.sum(0) - t_img.sum(0)).abs().mean())