import torch

with FakeTensorMode(shape_env=ShapeEnv()):
    y = torch.tensor([[1, 2], [3, 4]])
    x = torch.tensor([2])
    y.repeat_interleave(x)