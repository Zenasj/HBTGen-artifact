import torch.nn as nn

3
import torch, onnxruntime
print(torch.__version__, onnxruntime.__version__)
class M(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x)
torch.onnx.export(M(), torch.rand(3,4), "debug.onnx", opset_version=11)
onnxruntime.InferenceSession("debug.onnx")