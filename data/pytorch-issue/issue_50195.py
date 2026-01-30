import torch
import torch.nn as nn

@torch.jit.script
def func(x):
    return x * 0.5 * (1.0 + torch.erf(x))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input):
        return func(input)

model = Model()
model.eval()

a = torch.ones(32, 32, dtype=torch.float32)
torch.onnx.export(model, a, 'th_func.onnx', verbose=False, opset_version=11)

import onnxruntime
sess = onnxruntime.InferenceSession('th_func.onnx')

out = sess.run(None, {sess.get_inputs()[0].name: a.numpy()})
print(model(a))
print(out)