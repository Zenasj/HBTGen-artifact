import torch
import torch.nn as nn

class Reproducer(nn.Module):
    def forward(self, x):
        dim = x.size(0)
        mask = torch.zeros(x.size())
        mask.fill_(0) # error generated here
        return x * mask

model = Reproducer()

torch.onnx.export(
    model,
    (torch.zeros(3, 3),),
    "reproducer.onnx",
    input_names=["x"],
    output_names=["out"],
    verbose=True
)

import torch
import torch.nn as nn

class Reproducer(nn.Module):
    def forward(self, x):
        dim = x.size(0)
        fill_value = 1e4
        mask = torch.ones(x.size()) * fill_value
        return x * mask

model = Reproducer()

torch.onnx.export(
    model,
    (torch.zeros(3, 3),),
    "reproducer.onnx",
    input_names=["x"],
    output_names=["out"],
    verbose=True
)