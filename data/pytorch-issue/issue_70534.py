import torch.nn as nn
import random

import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


x_np = np.random.rand(1, 5, 8, 16).astype(np.float32)
y_np = np.random.rand(3, 1, 16, 4).astype(np.float32)
filename = 'output.onnx'

dummy_inputs = [torch.tensor(x_np), torch.tensor(y_np)]
print(Model()(*dummy_inputs).shape)
torch.onnx.export(
    Model(), tuple(dummy_inputs),
    filename,
    input_names=['x', 'y'],
    opset_version=14)

# uncomment the following lines to see what onnx shape inferencer complains about the shapes conflict
# from onnx import shape_inference, load
# inferred_model = shape_inference.infer_shapes(load("output.onnx"))