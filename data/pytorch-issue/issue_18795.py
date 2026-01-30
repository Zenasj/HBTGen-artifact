import numpy as np

import torch.nn as nn
import torch.onnx

class MinLayer(nn.Module):
    def __init__(self):
        super(MinLayer, self).__init__()

    def forward(self, inp):
        return inp.min()
        # return inp.max()
        # return inp.ceil()
        # return torch.min(inp)
        # return torch.max(inp)
        # return torch.ceil(inp)

model = MinLayer()
dummy_input = torch.zeros(1)
torch.onnx.export(model, dummy_input, '/tmp/tmp.onnx', verbose=True)