# Redefine `mul` at torch/onnx/symbolic_opset9.py
def mul(g, self, other):
    raise RuntimeError("c3p0 bug bug")