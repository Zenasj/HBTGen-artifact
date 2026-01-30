import torch.nn.functional as F

import torch
from torch.nn import functional as F

def grid_sample_v2(input, grid):
    # grid: [-1, 1]
    N, C, H, W = input.shape
    gridx = grid[:, :, :, 0]
    gridy = grid[:, :, :, 1]
    gridx = ((gridx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
    gridy = ((gridy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
    newgrid = torch.stack([gridx, gridy], dim=-1)
    return F.grid_sample(input, newgrid)

N = 500
input = torch.rand(N, 1, 32, 74)
input2 = F.interpolate(input, scale_factor=5, align_corners=False, mode='bilinear')

grid = torch.rand(N, 18, 17, 2) * 1.8 - 0.9  # use coordinates in -0.9, 0.9 to avoid boundary effects

output = F.grid_sample(input, grid)
output2 = F.grid_sample(input2, grid)
diff = torch.abs(output2 - output)
print("DIFFv1:", diff.mean())

output = grid_sample_v2(input, grid)
output2 = grid_sample_v2(input2, grid)
diff = torch.abs(output2 - output)
print("DIFFv2:", diff.mean())