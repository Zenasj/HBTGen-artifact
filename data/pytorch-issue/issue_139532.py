import torch.nn as nn

import torch
import onnx

class Foo(torch.nn.Module):
    def forward(self, x):
        (a0, a1), (b0, b1), (c0, c1, c2) = x
        return a0 + a1 + b0 + b1 + c0 + c1 + c2

f = Foo()
inputs = (
    (1, 2),
    (
        torch.randn(4, 4),
        torch.randn(4, 4),
    ),
    (
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
    ),
)

input_names = ["a", "b", "c", "d", "e", "f", "g"]
dynamic_axes = {
    "c": {0: "c_dim_0", 1: "c_dim_1"},
    "e": {0: "e_dim_0", 1: "e_dim_1"},
    "f": {0: "f_dim_0", 1: "f_dim_1"},
}

torch.onnx.export(f, (inputs,), "nested.onnx", dynamic_axes=dynamic_axes, input_names=input_names, verbose=True)
onnx_model = onnx.load("nested.onnx")
print(onnx_model.graph.input)