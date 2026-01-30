import torch.nn as nn

py
import io
import torch
from torch import Tensor
import onnxruntime

def f() -> Tensor:
    mask = torch.zeros(100, dtype=torch.bool)
    indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
    mask[indices] = True  # offending line
    return mask

class Module(torch.nn.Module):
    def forward(self, *args, **kwargs):
        return f()

model = Module()
model.eval()

model()  # works fine

onnx_io = io.BytesIO()
torch.onnx.export(model, [], onnx_io, opset_version=11)

ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
ort_outs = ort_session.run(None, {})  # errors