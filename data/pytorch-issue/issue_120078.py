import torch
import torch.nn as nn

py
a = nn.Sequential(nn.PixelShuffle(2))
onnx_program = torch.onnx.dynamo_export(a, torch.rand(1, 8, 256, 256))