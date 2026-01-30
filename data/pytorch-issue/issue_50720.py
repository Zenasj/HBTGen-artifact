import torch
import torch.nn as nn
import numpy as np

inp = tensor(
    [[[[-0.3096,  0.3871, -0.8741, -1.2659, -0.0505],
       [-0.1814,  0.6141,  0.7843,  1.1278, -0.4891],
       [ 2.4613, -0.7179, -0.2446,  0.6929, -0.3115],
       [ 0.8652, -0.6757,  1.9093, -0.5734,  0.3151],
       [ 0.7885,  0.0279,  0.3433, -0.2992,  1.0210]]]])
weight = torch.linspace(-1.0, 1.0, 9).view(1, 1, 3, 3)
inp = torch.quantize_per_tensor(inp, scale=0.15, zero_point=120, dtype=torch.quint8)
weight = torch.quantize_per_tensor(weight, 0.0007, zero_point=0, dtype=torch.qint8)
qf_out = torch.nn.quantized.functional.conv2d(inp, weight, scale=0.0009, zero_point=121, bias=None, dtype=torch.quint8)
ref_out = torch.nn.functional.conv2d(inp.dequantize().cpu(), weight.dequantize().cpu(), bias=None)
ref_out = torch.quantize_per_tensor(ref_out, scale=0.0009, zero_point=121, dtype=torch.quint8)
print(qf_out.int_repr() - ref_out.int_repr())

tensor([[[[ 0,  0,  0],
          [15,  0,  0],
          [ 0,  0,  0]]]], dtype=torch.uint8)

inp = tensor(
    [[-1.1218, -1.8445, 1.0258, 2.1921],
     [-1.4322, 1.5293, 1.2118, 0.0423],
     [0.3807, 0.0538, 0.5047, -0.4987],
     [-0.4891, 0.0578, 0.9436, 0.9775]])
weight = tensor(
    [[-0.0047, -0.0042, -0.0057, -0.0080],
     [-0.0032, 0.0005, -0.0042, 0.0093],
     [-0.0013, -0.0040, -0.0099, -0.0027],
     [0.0089, -0.0031, 0.0037, 0.0070],
     [-0.0070, -0.0065, 0.0062, 0.0049]])
inp = torch.quantize_per_tensor(inp, scale=0.0158, zero_point=117, dtype=torch.quint8)
weight = torch.quantize_per_tensor(weight, scale=0.000078, zero_point=0, dtype=torch.qint8)
qf_out = torch.nn.quantized.functional.linear(inp, weight, scale=0.00021, zero_point=79, bias=None)
ref_out = torch.nn.functional.linear(inp.dequantize().cpu(), weight.dequantize().cpu(), bias=None)
ref_out = torch.quantize_per_tensor(ref_out, scale=0.00021, zero_point=79, dtype=torch.quint8)
print(qf_out.int_repr() - ref_out.int_repr())

tensor([[40,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]], dtype=torch.uint8)