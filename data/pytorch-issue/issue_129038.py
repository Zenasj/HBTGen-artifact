import torch.nn as nn

import torch
import torch.export._trace
from torch import nn
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)

    def forward(self, x):
        x = self.conv(x)  # not neccesary to have a conv to reproduce the bug
        x = torch.cat([x, x])
        return x

model = Net()
quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
model = torch.export._trace._export(
    model, (torch.rand(1, 3, 32, 32),), pre_dispatch=True
).module()
model = prepare_pt2e(model, quantizer)