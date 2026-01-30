import torch.nn as nn

import onnx
import onnx.numpy_helper
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return x.expand((-1, 3, 3, 1))

model = Model()
torch.onnx.export(model, torch.rand(3, 3, 1, 1), 'expand.onnx')

m = onnx.load('expand.onnx')
n = m.graph.node[0]
assert n.op_type == 'Constant'
# [-1  3  3  1]
print(onnx.numpy_helper.to_array(n.attribute[0].t))