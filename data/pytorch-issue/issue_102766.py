import torch

mod = torch.compile(mod)
mod.is_compiled = True
assert "is_compiled" in dir(mod)