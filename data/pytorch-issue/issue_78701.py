import torch.nn as nn

import torch
import onnx
import numpy as np


class Model(torch.nn.Module):
    def forward(self, x):
        return torch.log2(x)


x = torch.tensor(1.)  # scalar
model = Model()
torch.onnx.export(model, (x, ), "output.onnx", opset_version=14,
                  output_names=['o0'], input_names=['i0'])
y_trh = model(x).numpy()

model = onnx.load("output.onnx")
print(model.graph.output[0])


import onnxruntime as ort
sess = ort.InferenceSession(
    "output.onnx", providers=['CPUExecutionProvider'])
y_ort = sess.run(['o0'], {'i0': x.numpy()})[0]
assert y_ort.shape == y_trh.shape, 'shape mismatch, ORT is `{}` but PyTorch is `{}`'.format(
    y_ort.shape, y_trh.shape)