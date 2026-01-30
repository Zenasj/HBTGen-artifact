import torch.nn as nn

import torch


class M(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias


class N(torch.nn.Module):
    def __init__(self, layers: int = 3):
        super().__init__()
        # 'bias' is same value for all layers, hence common sub expression.
        self.layers = torch.nn.ModuleList(
            [M(bias=torch.tensor([1.0])) for i in range(layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


x = torch.randn(8192, 1, 1)
model = N()

torch.onnx.export(
    model,
    x,
    "model.onnx",
    verbose=True,
    opset_version=15,
    input_names=["x"],
    dynamic_axes={"x": [0]},
)