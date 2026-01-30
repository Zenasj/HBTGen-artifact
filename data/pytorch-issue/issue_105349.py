import torch.nn as nn

import torch
 
# export works fine if divisor_override=None, but fails when divisor_override is integer value
model = torch.nn.AvgPool2d(kernel_size=5, stride=2, padding=1, divisor_override=3)
model.eval()
 
rand_inp = torch.randn(1, 3, 8, 8)
 
torch.onnx.export(model, rand_inp, "AvgPool2dModel.onnx", verbose=True)