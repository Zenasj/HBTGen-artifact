import torch
import torch.nn as nn


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

  def forward(self, theta, size):
    return torch.nn.functional.affine_grid(theta, size, align_corners=None)


model = Model()
theta = torch.ones((1, 2, 3))
size = torch.Size((1,3,24,24))
ep = torch.export.export(model, (theta, size,), strict=False)

args, kwargs = ep.example_inputs

# Fail with TreeSpec error
ep.module()(*args)

# Pass
from torch.utils._pytree import tree_map_only
args = tree_map_only(
    torch.Size,
    lambda x: torch.Tensor(x), 
    args
)
ep.module()(*args)