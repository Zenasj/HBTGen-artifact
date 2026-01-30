import torch.nn as nn
import torchvision

import torch
from torch.onnx import utils

class SkipConnectionModule(torch.nn.Module):
    def forward(self, x):
        out = x
        out += x
        out = torch.nn.functional.relu(out, inplace=True)

module = SkipConnectionModule()
x = torch.randn(4, 4)
_, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=13)
print(torch.__version__, unconvertible_ops)

import torch
from torch.onnx import utils
from torchvision.models.resnet import BasicBlock

module = BasicBlock(3, 3)
x = torch.randn(1, 3, 4, 4)
_, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=13)
print(torch.__version__, unconvertible_ops)

import torch
from torch.onnx import utils

class SkipConnectionModule(torch.nn.Module):
    def forward(self, x):
        out = x
        out += x
        out = torch.nn.functional.relu(out, inplace=True)
        return out

module = SkipConnectionModule()
x = torch.randn(4, 4)
_, unconvertible_ops = utils.unconvertible_ops(module, (x,), opset_version=13)
print(torch.__version__, unconvertible_ops)