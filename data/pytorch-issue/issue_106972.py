import torch.nn as nn

import torch

class MySqueeze(torch.autograd.Function):

    @staticmethod
    def symbolic(g, input, dim):
        return g.op('Squeeze', input, axes_i=[dim])

    @staticmethod
    def forward(ctx, input, dim):
        return input.squeeze(dim)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim):
        return MySqueeze.apply(x, dim)

input = torch.randn(1, 3)
dim = torch.tensor(0)

model = Model()
print(model(input, dim))

with torch.no_grad():
    input = torch.randn(1, 3)
    dim = torch.tensor(0)
    torch.onnx.export(
        model, 
        (input, dim), 
        'model.onnx',
)

torch.onnx.export(
    model,
    (input, dim),
    'model.onnx',
    opset_version=14,
    autograd_inlining=False
)