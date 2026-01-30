import torch.nn as nn
import random

import numpy as np
import onnx
import onnxruntime as ort
import torch


class MHAWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mha1 = torch.nn.MultiheadAttention(512, 8, batch_first=False)
    def forward(self, x, mask):
        return self.mha1(x, x, x, attn_mask=mask, need_weights=False)[0]

mha = MHAWrapper()
x = torch.rand(size=(62, 1, 512))
mask = torch.randint(0, 2, size=(62, 62), dtype=bool)

torch.onnx.export(
    mha,
    (x, mask),
    "mha.onnx",
    verbose=True,
    input_names=["x", "mask"],
    output_names=["out"],
    dynamic_axes={
        "x": {0: "num_elem"},
        "mask": {0: "num_elem", 1: "num_elem"},
        "out": {0: "num_elem"}
    }
)

onnx_model = onnx.load("./mha.onnx")
onnx.checker.check_model(onnx_model)

onnx_model = onnx.load("/root/gnn-diffusion/mha.onnx")
onnx.checker.check_model(onnx_model)

ort_sess = ort.InferenceSession('/root/gnn-diffusion/mha.onnx')
outputs = ort_sess.run(
    None,
    {
        "x": np.random.rand(83, 1, 512).astype(np.float32),
        "mask": np.random.randint(0, 2, size=(83, 83), dtype=bool),
    }
)

import numpy as np
import onnx
import onnxruntime as ort
import torch


class MHAWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mha1 = torch.nn.MultiheadAttention(512, 8, batch_first=False)
    def forward(self, x, mask):
        return self.mha1(x, x, x, attn_mask=mask, need_weights=False)[0]

mha = MHAWrapper()
x = torch.rand(size=(62, 1, 512))
mask = torch.randint(0, 2, size=(62, 62), dtype=bool)

onnx_program = torch.onnx.export(
    mha,
    (x, mask),
    input_names=["x", "mask"],
    output_names=["out"],
    dynamic_axes={
        "x": {0: "num_elem"},
        "mask": {0: "num_elem", 1: "num_elem"},
        "out": {0: "num_elem"}
    },
    dynamo=True
)

onnx_program.optimize()
onnx_program.save("mha.onnx")

onnx_model = onnx.load("mha.onnx")
onnx.checker.check_model(onnx_model)
ort_sess = ort.InferenceSession('mha.onnx')
outputs = ort_sess.run(
    None,
    {
        "x": np.random.rand(83, 1, 512).astype(np.float32),
        "mask": np.random.randint(0, 2, size=(83, 83), dtype=bool),
    }
)
print(outputs[0].shape)