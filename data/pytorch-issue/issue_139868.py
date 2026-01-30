import torch.nn as nn

import torch
from torch import nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Conv2d(3, 64, 3, bias=True)
    
    def forward(self, x: torch.Tensor):
        x = self.fc0(x)
        return x[:, 2:, :-1] # failed
        # return x[:, :, :, :2] # failed
        # return x[:, 2:] # ok
        # return x[:, :-1] # ok
        # return x[:, :, 2:, 2:] # ok
        # return x[:, :, 2:, :-1] # failed

model = Model()
tensor_x = torch.rand((1, 3, 64,64), dtype=torch.float32)
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(model, tensor_x, export_options=export_options)