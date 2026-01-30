import torch

def aten_linalg_inv(g, arg):
    return g.op("com.microsoft::Inverse", arg)


# Register custom symbolic function
torch.onnx.register_custom_op_symbolic("aten::linalg_inv", aten_linalg_inv, 17)

def aten_linalg_lstsq(g, a, b, **_):
    return g.op("MatMul", g.op("com.microsoft::Inverse", a), b)

torch.onnx.register_custom_op_symbolic("aten::linalg_lstsq", aten_linalg_lstsq, 17)

def aten_linalg_lstsq(g, a, b, **_):
    return g.op("your.domain::Lstsq", a, b)

torch.onnx.register_custom_op_symbolic("aten::linalg_lstsq", aten_linalg_lstsq, 17)