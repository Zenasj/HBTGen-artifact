import torch.nn as nn

import torch
from torch import Tensor
from typing import Tuple

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self) -> Tuple[Tensor, Tensor]:
        a = (torch.zeros(5), torch.zeros(5),)
        for x in range(5):
            c, d = a
            a = (c, d)
        return a

scripted_module = torch.jit.script(MyModule())

torch.onnx.export(
    scripted_module,
    tuple(),
    "./model.onnx",
    do_constant_folding=True,
    input_names=[],
    output_names=["OUTPUT__0", "OUTPUT__1"],
    opset_version=13,
    example_outputs=(torch.zeros(5), torch.zeros(5),)
)