import torch.nn as nn

from torch import nn
import torch.nn.functional as F

import torch
import torch.onnx


def match_size_2d(t, sized):
    assert t.dim() == 4 and sized.dim() == 4
    dh = sized.size(2) - t.size(2)
    dw = sized.size(3) - t.size(3)

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(t, pad)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        x = F.avg_pool2d(input, 2, ceil_mode=True)  
        upscaled = F.interpolate(x, scale_factor=2)

        # input is 51x51, upscaled is 52x52, trim to match
        upscaled = match_size_2d(upscaled, input) 
                
        return torch.cat([input, upscaled], dim=1)

model = MyModel()

dummy = torch.ones((1, 64, 51, 51))
out = model(dummy)
torch.onnx.export(model,               # model being run
                  dummy,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # wether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input_1':{0:'batch'}})

@torch.jit.script
def trim_2d(t, sized):
    return t[:, :, :sized.size(2), :sized.size(3)]

@torch.jit.script
def trim_2d(t, sized):
    return t.narrow(2, 0, sized.size(2)).narrow(3, 0, sized.size(3))