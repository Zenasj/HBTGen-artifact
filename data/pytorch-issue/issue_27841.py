import torch.nn as nn

from torch import nn
import torch.nn.functional as F
import onnx
import torch
import torch.onnx

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):

        x = F.avg_pool2d(input, 2, ceil_mode=True)  
        upscaled = F.interpolate(x, scale_factor=2)
        return  torch.cat([input, upscaled], dim=1)

model = MyModel()

dummy = torch.ones((1, 64, 50, 50))
out = model(dummy)
torch.onnx.export(model,               # model being run
                  dummy,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # wether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input':{0:'batch'}})


model = onnx.load("model.onnx")
onnx.checker.check_model(model)
onnx.helper.printable_graph(model.graph)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x = F.avg_pool2d(input, 2)
        y = F.avg_pool2d(input, 2)
        return x + y