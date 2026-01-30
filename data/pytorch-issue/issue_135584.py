import torch.nn as nn

import os

os.environ["TORCH_COMPILE_DEBUG"] = "1"

import torch


class Model(torch.nn.Module):
    def forward(self, x, ks0):
        return x.sum()


example_inputs = (
    torch.tensor([0, 3, 6], device="cuda", dtype=torch.int64),
    70,
)
_ = torch._export.aot_compile(
    Model(),
    example_inputs,
)
print("done")