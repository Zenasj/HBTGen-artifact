import torch.nn as nn

import torch
from torch import nn
from functools import partial
class test(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.0009)
        self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

model = test(8, 16)
model = model.cuda().eval()
with torch.inference_mode():
    inputs = torch.randn(2,8,64,64, device='cuda')
    # Export the model
    torch.onnx.export(model,               # model being run
                    inputs,                         # model input (or a tuple for multiple inputs)
                    "model.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=16,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    dynamic_axes = {'input_0':{3: "x", 2: "y"}, 'output_0':{3: "x", 2: "y"}},
                    input_names = ['input_0'],
                    output_names = ['output_0'])