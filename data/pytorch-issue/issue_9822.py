class MyModel(nn.Module):
    def forward(self, x):
        return x[:, :, :, 1:-1]

import torch
import torch.onnx as tonnx
import torch.nn as nn
from torch.autograd import Variable

class MyModel(nn.Module):
    def forward(self, x):
        return x[:, :, :, 1:-1]

dummy_input = Variable(torch.randn(1, 3, 224, 224))
model = MyModel()

tonnx.export(model, dummy_input, "/tmp/model.onnx", verbose=True)