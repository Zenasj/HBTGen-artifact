import torch.nn as nn

import torch

class Module(torch.nn.Module):
    def forward(self, x):
        return x.expand(x.shape[0], -1, -1)  # Crashes here during ONNX export

model = Module()
dummy_inputs = tuple(torch.randn(1, 1, 192))

# Running the model works fine
res = model(*dummy_inputs) 

# Exporting to ONNX causes core dump
torch.onnx.export(model, opset_version=20, f="./m.onnx", args=dummy_inputs)