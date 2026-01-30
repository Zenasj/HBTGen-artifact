import torch
import torch.nn as nn

x = torch.ones(20, 16, 50, 40, requires_grad=True)
import io
f = io.BytesIO()
torch.onnx.export(nn.Conv2d(16, 13, 3, bias=False), x, f,
                        keep_initializers_as_inputs=False, verbose=True, opset_version=14)
import onnx
f.seek(0)
print(onnx.load(f))