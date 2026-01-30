import torch.nn as nn

import torch
import torch.onnx
import torch.nn.functional as F

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(231, 77))
        self.b = torch.nn.Parameter(torch.randn(231))

    def forward(self, x):
        q, k, v = x, x, x
        q, k, v = F._in_projection_packed(q, k, v, self.w, self.b)
        return q + k + v

model = SimpleModel()

example_input = torch.randint(0, 11, (1, 77), dtype=torch.float32)

script_module = torch.jit.trace(model, example_input)

torch.onnx.export(script_module, example_input, "in_proj.onnx", input_names=['input'], output_names=['output'])