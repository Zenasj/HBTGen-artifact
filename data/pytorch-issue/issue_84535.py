import torch
import torch.nn as nn

x = torch.rand((128, 32, 1, 32, 1, 1))
assert torch.nn.functional.pad(x, [0, 0, 0, 0, 0, 0, 0, 0]).shape == x.shape

x = torch.rand((128, 32, 1, 32, 1, 1)).to("mps")
assert torch.nn.functional.pad(x, [0, 0, 0, 0, 0, 0, 0, 0]).shape == x.shape