import torch.nn as nn

py
import torch

class Net(torch.nn.Module):
    def forward(self, x):
        # return x.bernoulli(0.2)  # case #1
        # return x.bernoulli_()    # case #2
        # return x.bernoulli_(0.2) # case #3
        return x.bernoulli()       # case #4

model = Net()
x = torch.rand(4, 5)
torch.onnx.export(model, x, 'bernoulli_export.onnx', opset_version=15)