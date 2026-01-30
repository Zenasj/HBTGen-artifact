import torch.nn as nn

import torch
import torch_tensorrt
class TestChunk(torch.nn.Module):
    def forward(self, input):
        out = torch.ops.aten.chunk.default(input, 3, 0)
        return out

inputs = [torch.randn(3)]
inputs_zero_shape = torch.export.Dim("shape", min=1, max=3)
dynamic_shapes = [[torch.export.Dim("shape", min=1, max=3)]]
exp_program = torch.export.export(TestChunk(), tuple(inputs), dynamic_shapes=dynamic_shapes)
trt_gm = torch_tensorrt.dynamo.compile(exp_program, inputs)
# Run inference
trt_gm(*inputs)