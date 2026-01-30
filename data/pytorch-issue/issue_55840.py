import torch.nn as nn

import torch

dummy_input = torch.randn(4, 384, 2, device='cuda')
input_names = ["logits"]

class Split(torch.nn.Module):
    def forward(self, x):
        return x.split(1, dim=-1)
model = Split()

with torch.no_grad():
    torch.onnx.export(
        model, dummy_input, "split.onnx", verbose=True, opset_version=13,
        enable_onnx_checker=True, do_constant_folding=False,
        input_names=input_names,
        dynamic_axes={input_names[0]: {0: 'batch'}}
    )