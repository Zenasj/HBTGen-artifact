import torch
from torch import nn
class Foo(nn.Module):
    def forward(self, x, y):
        ori_dtype = y.dtype
        x = x.view(torch.int32)
        y = y.view(torch.int32)
        rst = (~x | y)
        rst = rst.view(ori_dtype)
        return rst

model = Foo()
data_in = (torch.randn(10), torch.randn(10))
torch.onnx.export(model, data_in, 'tmp.onnx')

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxscript
import onnxruntime
from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print(torch.__version__)
print(onnxscript.__version__)
print(onnxruntime.__version__)

torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("my_image_classifier.onnx")

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxscript
import onnxruntime
from onnxscript import opset18  # opset 18 is the latest (and only) supported version for now


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print(torch.__version__)
print(onnxscript.__version__)
print(onnxruntime.__version__)

torch_model = MyModel()
torch_input = torch.randn(1, 1, 32, 32)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input)
onnx_program.save("my_image_classifier.onnx")

import torch
from torch import nn
class Foo(nn.Module):
    def forward(self, x, y):
        ori_dtype = y.dtype
        x = x.view(torch.int32)
        y = y.view(torch.int32)
        rst = (~x | y)
        rst = rst.view(ori_dtype)
        return rst

model = Foo()
data_in = (torch.randn(10), torch.randn(10))
torch.onnx.dynamo_export(model, data_in, 'tmp.onnx')