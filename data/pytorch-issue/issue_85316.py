import torch.nn as nn

import torch
from io import BytesIO
import onnx


class MulLeakyReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        return x * self.m(x)


f = BytesIO()
torch.onnx.export(
    MulLeakyReLU(),
    torch.randn((), dtype=torch.float64),
    f,
    verbose=True,
    opset_version=14,
)
onnx_model = onnx.load_from_string(f.getvalue())
onnx.checker.check_model(onnx_model, full_check=True)
assert MulLeakyReLU()(torch.randn((), dtype=torch.float64)).dtype == torch.float64
onnx_output_type = onnx_model.graph.output[0].type.tensor_type.elem_type
assert (
    onnx_output_type == onnx.TensorProto.DataType.DOUBLE
), f"Expected output to be double but the converted ONNX model outputs is {onnx.TensorProto.DataType.Name(onnx_output_type)}"

"""
Exported graph: graph(%input : Double(requires_grad=0, device=cpu)):
  %/m/LeakyRelu_output_0 : Double(requires_grad=0, device=cpu) = onnx::LeakyRelu[alpha=0.10000000000000001, onnx_name="/m/LeakyRelu"](%input), scope: __main__.MulLeakyReLU::/torch.nn.modules.activation.LeakyReLU::m
  %/Cast_output_0 : Float(requires_grad=0, device=cpu) = onnx::Cast[to=1, onnx_name="/Cast"](%input), scope: __main__.MulLeakyReLU:: # test.py:14:0
  %/Cast_1_output_0 : Float(requires_grad=0, device=cpu) = onnx::Cast[to=1, onnx_name="/Cast_1"](%/m/LeakyRelu_output_0), scope: __main__.MulLeakyReLU:: # test.py:14:0
  %5 : Float(requires_grad=0, device=cpu) = onnx::Mul[onnx_name="/Mul"](%/Cast_output_0, %/Cast_1_output_0), scope: __main__.MulLeakyReLU:: # test.py:14:0
  return (%5)

Traceback (most recent call last):
  File "test.py", line 29, in <module>
    assert (
AssertionError: Expected output to be double but the converted ONNX model outputs is FLOAT
"""

torch.randn((), dtype=torch.float64),