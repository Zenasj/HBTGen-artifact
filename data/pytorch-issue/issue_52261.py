import torch
import torch.nn as nn

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)

    def forward(self, x):
        return self.conv1(x)

import os
import torch.onnx
from torch.onnx import OperatorExportTypes
torch.onnx.export(SimpleConv(),
                  torch.zeros((1,1,16,16)),
                  open(os.devnull,"wb"),
                  verbose=True,
                  operator_export_type=OperatorExportTypes.RAW
                 )