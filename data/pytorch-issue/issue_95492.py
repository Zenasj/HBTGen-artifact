import torch.nn.functional as F

import torch
import torch.nn as nn

from torch.ao.quantization import get_default_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.nn import functional as F


class myConv2d_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input):
        return F.conv2d(input, self.weight, None, [1, 1], [0, 0], [1, 1], 1)

class myConv2d_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input):
        return F.conv2d(input, self.weight, bias=None, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1)

# This works
#model = myConv2d_1(8, 64, 3)

# This doesn't work
model = myConv2d_2(8, 64, 3)

model.eval()
qconfig = get_default_qconfig("fbgemm")
qconfig_mapping = QConfigMapping().set_global(qconfig)

example_inputs = torch.randn(1, 8, 224, 224)

prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
model(example_inputs)
quantized_model = convert_fx(prepared_model)
out_q = quantized_model(example_inputs)
quantized_model.print_readable()