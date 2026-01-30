import torch.nn as nn
import torch.nn.functional as F
import random

import torch

F = torch.nn.functional

def apply_grid_sample(imgs, grid, upcast=False, align_corners=False):
    if upcast:
        imgs = imgs.to(torch.float32)
        grid = grid.to(torch.float32)

    result = F.grid_sample(imgs, grid, align_corners=align_corners)

    return result.to(torch.float32)


N = 1
align_corners = False

source_size = 192
target_size = 112

source_shape = N, 3, source_size, source_size
target_shape = N, 3, target_size, target_size

torch.random.manual_seed(1)
imgs = torch.randn(*source_shape)
imgs = imgs.to(dtype=torch.float16, device="cuda")

identity = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
grid = F.affine_grid(theta=identity, size=target_shape, align_corners=align_corners)
grid = grid.to(dtype=torch.float16, device="cuda")

full_precision = apply_grid_sample(imgs, grid, upcast=True, align_corners=align_corners)
half_precision = apply_grid_sample(imgs, grid, upcast=False, align_corners=align_corners)

abs_full = abs(full_precision)
abs_full_mean = abs_full.mean()
abs_full_max = abs_full.max()
print(f"Expected result has mean absolute value of {abs_full_mean:.4f} and maximum absolute value of {abs_full_max:.4f}.")

abs_error = abs(full_precision - half_precision)
abs_error_mean = abs_error.mean()
abs_error_max = abs_error.max()
print(f"Half precision result is off by {abs_error_mean:.4f} ({abs_error_mean/abs_full_mean:.2%}) on average and {abs_error_max:.4f} ({abs_error_max/abs_full_max:.2%}) at maximum.")