import torch.nn as nn

two = torch.tensor(2.0, dtype=torch.float32)

import difflib

import onnx
import torch


@torch.jit.script
def add_with_zero_dim_tensor(x):
    if x.numel() <= 0:
        return torch.empty((1,), dtype=torch.float32)
    else:
        x_max = x.max()
        y = x_max + torch.tensor(1, dtype=torch.float32)
        return y * 2

@torch.jit.script
def add_with_one_dim_tensor(x):
    if x.numel() <= 0:
        return torch.empty((1,), dtype=torch.float32)
    else:
        x_max = x.max()
        y = x_max + torch.tensor([1], dtype=torch.float32)
        return y * 2

class AddWithZeroDimTensor(torch.nn.Module):

    def forward(self, x):
        return add_with_zero_dim_tensor(x)

class AddWithOneDimTensor(torch.nn.Module):

    def forward(self, x):
        return add_with_one_dim_tensor(x)

def trace_and_print(model, x):
    script = torch.jit.trace(model, x)
    print(script.graph)

def export_to_onnx_and_print(model, x):
    outname = f'{model.__class__.__name__}.onnx'
    torch.onnx.export(model, x, outname, input_names=['x'])
    return str(onnx.load(outname)).split('\n')

x = torch.ones((10,), dtype=torch.float32)

s1 = export_to_onnx_and_print(AddWithZeroDimTensor(), x)
s2 = export_to_onnx_and_print(AddWithOneDimTensor(), x)
print('\n'.join(difflib.Differ().compare(s1, s2)))