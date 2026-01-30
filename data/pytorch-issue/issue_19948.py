import torch.nn as nn

import torch

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()

    def forward(self, x):
        a = x.repeat([3,2])
        a[:2] = x*2
        return a

onnx_path = "slice_assign.onnx"
model = TestModule()
torch_in = (torch.randint(0, 10, (2, 1)), )
torch.onnx.export(model, torch_in, onnx_path, verbose=True)