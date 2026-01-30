import torch.nn as nn

import torch


class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1


model = Model().eval()
input = torch.randn(10)

dim = torch.export.Dim("dim_0")
dim_even = 2 * dim

exported_program = torch.export.export(
    model,
    args=(input,),
    dynamic_shapes=({0: dim_even},),
)
torch._inductor.aoti_compile_and_package(exported_program)

dim = torch.export.Dim("dim_0")

exported_program = torch.export.export(
    model,
    args=(input,),
    dynamic_shapes=({0: dim},),
)

from __future__ import annotations

import torch


class Model(torch.nn.Module):
    def forward(self, x, y):
        return x + 1, y + 1


model = Model().eval()
input_0 = torch.randn(10)
input_1 = torch.randn(6)

dim = torch.export.Dim("dim_0")
dim_even = 2 * dim
dim_plus_1 = dim + 1

exported_program = torch.export.export(
    model,
    args=(input_0, input_1),
    dynamic_shapes=({0: dim_even}, {0: dim_plus_1}),
)


input_0 = torch.randn(10)
input_1 = torch.randn(7)
exported_program.module()(input_0, input_1)