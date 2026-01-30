import torch.nn as nn

import torch
import torch.nn.functional as F

theta = torch.randn(4, 2, 3)
size = torch.Size((1, 1, 5, 5))
align_corners = False

def affine_grid_func(theta):
    return F.affine_grid(theta, size, align_corners=align_corners)

tangent_vector = torch.ones_like(theta)
output, jvp = torch.func.jvp(affine_grid_func, (theta,), (tangent_vector,))