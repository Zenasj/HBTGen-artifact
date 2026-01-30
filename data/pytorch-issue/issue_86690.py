import torch.nn as nn
import random

import torch
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np


class PadModel(torch.nn.Module):
    def forward(self, x):
        out = F.pad(x, (1, 1, 2, 2), "circular")
        return out

model = PadModel()

inputs = torch.randn(1, 1, 4, 4)
print("Exporting model")
torch.onnx.export(
    model,
    inputs,
    "pad_model.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['sample'],
    output_names=['output_0'])

ort_session = ort.InferenceSession("pad_model.onnx")

sample = np.random.randn(1, 1, 4, 4).astype(np.float32)
print("sample", sample.shape)
print(sample)
outputs = ort_session.run(
    None,
    {"sample": sample},
)

print("ONNX model output")
print(outputs[0])

print("PyTorch model output")
print(model(torch.Tensor(sample)))