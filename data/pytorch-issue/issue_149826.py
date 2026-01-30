import torch.nn as nn

import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, size):
        y = torch.nn.functional.interpolate(x, size=size.tolist())
        return y


model = Model()
x = torch.rand(1, 3, 400, 500)
size = torch.tensor([1024, 1024]).to(torch.int32)
y = model(x, size)

onnx_model = torch.onnx.export(model, (x, size), dynamo=True)

import torch


class CustomOp(torch.nn.Module):
    def forward(self, x: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        val = torch.onnx.ops.symbolic(
            "Resize", # Uses onnx::Resize op
            [x, torch.tensor([]), torch.tensor([]), size],
            {},
            dtype=x.dtype,
            shape=x.shape,
            version=1,
        )
        return val


model = CustomOp()
x = torch.rand(1, 3, 400, 500)
size = torch.tensor([1, 3, 1024, 1024]).to(torch.int64)
y = model(x, size)

onnx_model = torch.onnx.export(model, (x, size), dynamo=True)
size_new = torch.tensor([1, 3, 50, 50])
print(onxx_model(x, size_new)[0].shape)
# torch.Size([1, 3, 50, 50])