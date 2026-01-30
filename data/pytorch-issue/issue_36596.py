import torch
import torch.nn as nn

@torch.jit.script
def f2(self_min_size, self_max_size):
    # type: (int, int) -> Tensor
    scale_factor = 2.5
    if self_min_size * scale_factor > self_max_size:
        scale_factor = self_max_size / self_min_size

    return torch.tensor(scale_factor)


class MyModel(torch.nn.Module):
  def forward(self, x):
    scale_factor = f2(10, 15)
    y = torch.nn.functional.interpolate(
        x, scale_factor=float(scale_factor), mode='bilinear', align_corners=False)[0]

    return y

class MyModel(torch.nn.Module):
  def forward(self, x):
    scale_factor = f2(10, 15)
    y = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

    return y