import torch

print("Pytorch version", torch.__version__)
x = torch.randn(2, 2, 2)
scales = (torch.randn(2) + 1) * 0.05
zero_points = torch.zeros(2).to(torch.long)
out = torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)