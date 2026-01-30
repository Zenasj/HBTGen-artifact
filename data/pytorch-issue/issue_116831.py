import torch.nn as nn

py
import torch
import torch_tensorrt
import unittest

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        mul = relu * 0.5
        return mul

input = torch.randn((1, 3, 224, 224), dtype=torch.float).to("cuda")
model = MyModule().eval().cuda()

compile_spec = {
        "inputs": [
            torch_tensorrt.Input(
                input.shape, dtype=torch.float, format=torch.contiguous_format
            )
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "torch_executed_ops": {"torch.ops.aten.convolution.default"},
    }

exp_program = torch_tensorrt.dynamo.trace(model, **compile_spec)
trt_gm = torch_tensorrt.dynamo.compile(exp_program, **compile_spec)
trt_exp_program = torch_tensorrt.dynamo.export(trt_gm, [input], ir="exported_program")

torch.export.save(trt_exp_program, "/tmp/trt.ep")
deser_trt_exp_program = torch.export.load("/tmp/trt.ep")
outputs_pyt = model(input)
outputs_trt = trt_exp_program(input)