import torch.nn as nn

import torch
class Test(torch.nn.Module):
    def forward(self, x):
        return x + 1

x = torch.ones(2,3,4,5)
torch.onnx.export(Test(), tuple((x,)), 'test.onnx', verbose=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input':[0, 2, 3], 'output':[0, 2, 3]})