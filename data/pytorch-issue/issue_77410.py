import torch.nn as nn

python
import torch

class ExampleDataSize(torch.nn.Module):
    def forward(self, a):
        s = a.data.size()
        t = 0
        for i in range(4):
            t += s[i]
        return t


class ExampleShape(torch.nn.Module):
    def forward(self, a):
        s = a.shape
        t = 0
        for i in range(4):
            t += s[i]
        return t

data_size_ex = ExampleDataSize()
shape_ex = ExampleShape()

dummy_input = torch.autograd.Variable(
        torch.randn(1, 3, 64, 128)
    )

torch.onnx.export(
    data_size_ex,
    dummy_input,
    "data_size_ex.onnx",
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=11,
    input_names=["data"],
    dynamic_axes={
        "data": [0, 1, 2, 3]
    },
    output_names=["output"],
)

torch.onnx.export(
    shape_ex,
    dummy_input,
    "shape_ex.onnx",
    verbose=True,
    keep_initializers_as_inputs=True,
    opset_version=11,
    input_names=["data"],
    dynamic_axes={
        "data": [0, 1, 2, 3]
    },
    output_names=["output"],
)