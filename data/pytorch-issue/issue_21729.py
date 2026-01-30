import torch.nn as nn

from torch import nn

import torch
import torch.nn.functional as F
import onnxruntime as rt

class Upsample(torch.nn.Module):
    def forward(self, x):
        #l = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=1, bias=True)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

m = Upsample()
v = torch.randn(1,3,128,128, dtype=torch.float32, requires_grad=False)

torch.onnx.export(m, v, "test.onnx")
sess = rt.InferenceSession("test.onnx")