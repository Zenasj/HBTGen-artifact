import torch.nn as nn

import torch
from torch.autograd import Variable
import torch.onnx
import onnx
# from onnx_tf.backend import prepare

dummy_input = Variable(torch.randn(10, 6, 224, 224))
model = torch.nn.BatchNorm2d(num_features=6)
torch.onnx.export(model, dummy_input, "flownet.onnx", verbose=True)

#no error here
onnx_model = onnx.load("flownet.onnx")
# Check that the IR is well formed
onnx.checker.check_model(onnx_model)