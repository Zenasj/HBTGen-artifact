import torch

mode = FakeTensorMode(inner=None)
x = torch.empty(2, 2, device="cpu")
with enable_torch_dispatch_mode(mode):
    x = mode.from_tensor(x)
    x.logical_not_()