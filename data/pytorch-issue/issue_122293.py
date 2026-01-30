import torch.nn as nn

import torch
import torch.nn.functional as F


def interpolate_model(input, scale_factor, mode, align_corners):
    return F.interpolate(
        input=input, scale_factor=scale_factor, mode=mode, align_corners=align_corners
    )


comp_model = torch.compile(interpolate_model)

inp1 = torch.randn(2, 1, 4, 5, 6)
out1 = comp_model(inp1, 2, "trilinear", True)

inp2 = torch.randn(2, 1, 8, 8, 8)
out2 = comp_model(inp2, 2, "trilinear", True)