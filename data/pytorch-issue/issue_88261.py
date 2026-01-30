import torch.nn as nn

import io
import torch
from torch.onnx._internal import diagnostics

class CustomAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y

    @staticmethod
    def symbolic(g, x, y):
        return g.op("custom::CustomAdd", x, y)

class M(torch.nn.Module):
    def forward(self, x):
        return CustomAdd.apply(x, x)

# trigger warning for missing shape inference.
# rule = diagnostics.rules.node_missing_onnx_shape_inference
torch.onnx.export(M(), torch.randn(3, 4), io.BytesIO())

diagnostics.engine.pretty_print(verbose=True, level=diagnostics.levels.WARNING)