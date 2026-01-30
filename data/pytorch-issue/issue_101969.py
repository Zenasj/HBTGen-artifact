import torch.nn as nn

import torch
from torch import nn

class Dummy(nn.Module):
    def forward(self, x):
        return x.logit()

model = Dummy()
torch.onnx.export(model, torch.randn(1, 3, 640, 640), "logit.onnx")

@_onnx_symbolic("aten::logit")
@_beartype.beartype
def logit(g: jit_utils.GraphContext, self, eps):
    return log(g, div(g, self, sub(g, symbolic_helper._if_scalar_type_as(torch.ones(1), self), self)))