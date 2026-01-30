import torch

import onnx
model.eval()
torch.onnx.export(
    model,
    torch.randn(1, 3, 384, 512),
    'zoedepth_n.onnx',
    opset_version=13,
)