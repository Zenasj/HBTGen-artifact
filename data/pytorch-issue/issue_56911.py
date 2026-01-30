import torch.nn as nn

import torch
from torch import Tensor
from typing import Tuple

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        if t < 0:
            raise Exception("Negative input")
        else:
            return torch.zeros(5), torch.zeros(5)


scripted_module = torch.jit.script(MyModule())

torch.onnx.export(
    scripted_module,
    (torch.zeros(5),),
    "./model.onnx",
    do_constant_folding=True,
    input_names=["INPUT__0"],
    output_names=["OUTPUT__0", "OUTPUT__1"],
    opset_version=13,
    example_outputs=(torch.zeros(5), torch.zeros(5))
)