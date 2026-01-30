import torch.nn as nn

import torch

mod = torch.nn.Linear(10, 10).eval().cuda()
# Fine
mod.weight.to(device="meta")
# Errors
with torch._dispatch.python.enable_python_dispatcher():
    mod.weight.to(device="meta")