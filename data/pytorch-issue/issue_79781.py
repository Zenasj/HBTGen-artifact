import torch.nn as nn

import torch
from typing import Dict

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        # print("mdg forward x is", x)
        if x.sum() > 0:
            # print("+ve")
            return x
        else:
            # print("-ve")
            return -x

class MyCell(torch.nn.Module):
    def __init__(self, dg):
        super(MyCell, self).__init__()
        self.dg = dg
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, d: Dict[ str, torch.Tensor ], xx: torch.Tensor, hh: torch.Tensor):
        # print("dd is", dd)
        # print("d is", d)
        # x = d["x"]
        x = xx
        # print("x is", x)
        h = d["h"]
        # h = hh
        # print("h is", h)
        new_h = torch.tanh(self.dg(self.linear(x)) + h)
        # print("new_h is", new_h)
        return new_h, new_h

x = torch.rand(3, 4)
xx = torch.rand(3, 4)
h = torch.rand(3, 4)
hh = torch.rand(3, 4)

scripted_gate = torch.jit.script(MyDecisionGate())
my_cell = MyCell(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print("gate code")
print(scripted_gate.code)
print("cell code")
print(scripted_cell.code)

print("export")
torch.onnx.export(scripted_cell, ({"x": x, "h": h}, xx, hh ), "test.onnx", verbose=True, input_names=[ "inputdict", "inpx", "inph" ], output_names=[ "output" ],
        opset_version=9)