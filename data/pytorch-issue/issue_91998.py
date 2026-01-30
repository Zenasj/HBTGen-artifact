import torch

@_beartype.beartype
def triu_14(g: jit_utils.GraphContext, self, diagonal, out=None):
    """Triu opset 14 as defined in torch/onnx/symbolic_function14.py"""
    return g.op("Trilu", self, diagonal, upper_i=1)
# or from torch.onnx.symbolic_opset14 import triu as triu_14

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("::triu", triu_14, 13)

# Your code as usual
torch.onnx.export(model, ...)