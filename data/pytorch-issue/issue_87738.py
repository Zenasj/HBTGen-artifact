import torch.nn as nn

py
import io
import onnxruntime
import torch
from torch.onnx import register_custom_op_symbolic, unregister_custom_op_symbolic, symbolic_helper

@symbolic_helper.parse_args("v", "s")
def my_gelu(g, self, approximate):
    # approximate must be none to use contrib implementation.
    assert approximate == "none"
    return g.op("com.microsoft::Gelu", self, approximate_s=approximate)

register_custom_op_symbolic('::gelu', my_gelu, 1)

class CustomGelu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="none")

x = torch.eye(2, 2)

with io.BytesIO() as f:
    torch.onnx.export(CustomGelu(), x, f, verbose=True)
    
unregister_custom_op_symbolic("::gelu", 1)

py
import io
import onnxruntime
import torch
from torch.onnx import register_custom_op_symbolic, unregister_custom_op_symbolic

def my_inverse(g, self):
    return g.op("com.microsoft::Inverse", self).setType(self.type())

register_custom_op_symbolic('::inverse', my_inverse, 1)

class CustomInverse(torch.nn.Module):
    def forward(self, x):
        return torch.inverse(x)

x = torch.eye(2, 2)

with io.BytesIO() as f:
    torch.onnx.export(CustomInverse(), x, f, verbose=True)

unregister_custom_op_symbolic("::inverse", 1)