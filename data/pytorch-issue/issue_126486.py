from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.onnx_types import TensorType
from onnxscript.onnx_opset import opset18 as op


@torch_op("aten::glu", trace_only=True)
def aten_glu(self: TensorType, dim: int = -1) -> TensorType:
    """glu(Tensor self, int dim=-1) -> Tensor"""

    first, second = op.Split(self, num_outputs=2, axis=dim)
    return op.Mul(first, op.Sigmoid(second))