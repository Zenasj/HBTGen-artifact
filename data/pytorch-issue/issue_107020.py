import torch
import torch.nn as nn

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

mod = TestModule()
inp = torch.rand(1)
out = mod(inp)
mod2 = torch.fx.symbolic_trace(mod, concrete_args=[inp])

so, _ = torch._export.aot_compile(mod2, tuple([inp]))
# 2nd time, it will return None
so, _ = torch._export.aot_compile(mod2, tuple([inp]))
assert so is not None  # FAIL