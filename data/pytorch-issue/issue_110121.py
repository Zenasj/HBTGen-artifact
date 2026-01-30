import torch.nn as nn

import torch
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch._dynamo.utils import detect_fake_mode
import unittest
from typing import Sequence

class Linear(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(5, 7)

    def forward(self, x):
        return self.linear(x)

def mini_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
):
    fake_mode = detect_fake_mode(sample_inputs)

    with unittest.mock.patch.object(
            fake_mode, "allow_non_fake_inputs", True
        ), fake_mode:

        return aot_export_joint_simple(gm, sample_inputs, trace_joint=False)


sample_inputs = [torch.rand((3, 4, 5)).cuda()]
model = Linear().eval().cuda()
optimized = torch.compile(model, backend=mini_backend)
optimized(*sample_inputs)