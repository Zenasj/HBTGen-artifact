import torch.nn as nn

py
import torch
import onnxruntime as ort

class Test(torch.nn.Module):
            
    def forward(self, x):
        a, b, c = x.shape
        x = x.reshape(a, -1)
        d, e = x.shape
        x = x.reshape(e, -1)
        x = x.to(torch.float32) / torch.tensor(2., dtype=torch.float32)
        return x

body = Test().eval()

data = torch.randn(3, 2, 1)
y = body(data)
model = torch.jit.script(body)

torch.onnx.export(model, (data,), 'tmp.onnx', verbose=True, input_names=['data'], opset_version=13, example_outputs=(y,))
session = ort.InferenceSession("tmp.onnx")

py
class Model(torch.nn.Module):
    def forward(self, x, y=None, z=None):
        if y is not None:
            return x + y
        if z is not None:
            return x + z
        return x
m = Model()
x = torch.randn(2, 3)
z = torch.randn(2, 3)

torch.onnx.export(model, (x, None, z), 'test.onnx')