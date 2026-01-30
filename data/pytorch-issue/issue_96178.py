py
import torch

torch.manual_seed(0)
t = torch.testing.make_tensor((10_000, 256), dtype=torch.uint8, device="cpu", high=256)

assert not (t == 255).any()
assert (torch.unique(t) == torch.arange(255, dtype=torch.uint8)).all()