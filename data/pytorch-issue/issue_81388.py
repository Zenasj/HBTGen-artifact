import torch.nn as nn

3
import torch
import torch.onnx.symbolic_helper
import torch.onnx.symbolic_registry
from typing import List

@torch.onnx.symbolic_helper.parse_args("v", "b", "f")
def pad_sequence(g, input, batch_first, padding_value):
    return g.op("custom::MySequencePad", input)

torch.onnx.symbolic_registry.register_op("pad_sequence", pad_sequence, "", 13)

class Model(torch.nn.Module):
    def forward(self, xs: List[torch.Tensor]):
        return torch._C._nn.pad_sequence(xs, True, 0.)

# model = Model()
model = torch.jit.script(Model())
args = ([torch.rand(2, 5), torch.rand(3, 5)],)
torch.onnx.export(model, args, "test.onnx", opset_version=13)