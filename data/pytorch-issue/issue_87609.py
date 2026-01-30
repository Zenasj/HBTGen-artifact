import torch.nn as nn

import torch
from io import BytesIO
import onnx


class M(torch.nn.Module):
    @torch.no_grad()
    def forward(self, x, y):
        return torch.max(x, y)


inputs = (
    torch.randn((2, 2, 1, 1), dtype=torch.float64),
    torch.randn((), dtype=torch.float32),
)
f = BytesIO()
torch.onnx.export(M(), inputs, f, opset_version=13)

onnx_model = onnx.load_from_string(f.getvalue())
onnx.checker.check_model(onnx_model, full_check=True)
assert M()(*inputs).dtype == torch.float64
print(onnx.helper.printable_graph(onnx_model.graph))

"""
graph torch_jit (
  %onnx::Max_0[DOUBLE, 2x2x1x1]
  %onnx::Max_1[FLOAT, scalar]
) {
  %2 = Max(%onnx::Max_0, %onnx::Max_1)
  return %2
}
"""

"""
graph torch_jit (
  %onnx::Min_0[DOUBLE, 2x2x1x1]
  %onnx::Cast_1[FLOAT, scalar]
) {
  %/Cast_output_0 = Cast[to = 11](%onnx::Cast_1)
  %3 = Min(%onnx::Min_0, %/Cast_output_0)
  return %3
}
"""