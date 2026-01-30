import torch
from torch.testing._internal.optests import opcheck

inputs = torch.rand(3, device="cuda")
opcheck(torch.ops.aten.histc.default, args=(inputs,), kwargs={"bins": 4, "min": 0, "max": 1})

import torch

inputs = torch.rand(3, device="cuda")
histc_opt = torch.compile(torch.histc, dynamic=True, fullgraph=True)
histc_opt(inputs, bins=4, min=0, max=1)