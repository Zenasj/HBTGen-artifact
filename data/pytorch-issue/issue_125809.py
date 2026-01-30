import torch

x = torch.randn(2, 2, 2, dtype=torch.bfloat16)
scales = (torch.randn(2) + 1) * 0.05
zero_points = torch.zeros(2).to(torch.int32)
output = torch.fake_quantize_per_channel_affine(x, scales, zero_points, 1, 0, 255)