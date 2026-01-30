import torch.nn as nn

import os

import onnx
import torch


class OnesLike(torch.nn.Module):
    def forward(self, x):
        return x + torch.ones_like(x)


# torch 1.10 or later cannot export this if we change 100 to 1000.
x = torch.ones(100, 1000, 1000)
torch.onnx.export(OnesLike(), x, 'ones_like.onnx',
                  do_constant_folding=False,
                  opset_version=13)

m = onnx.load("ones_like.onnx")

# 1.9: ['Shape', 'ConstantOfShape', 'Add'] 210
# 1.10: ['Constant', 'Add'] 400000191
# 1.11: ['Constant', 'Add'] 400000233
print([n.op_type for n in m.graph.node], os.stat("ones_like.onnx").st_size)