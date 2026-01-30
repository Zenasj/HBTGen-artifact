import torch.nn.functional as F

py
import torch
import torch.nn as nn

torch.manual_seed(420)


class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.batchnorm = nn.BatchNorm2d(num_features=5)
        self.conv_weight = torch.randn(5, 3, 3, 3)
        self.conv_bias = torch.randn(5)

    def forward(self, x):
        self.conv.weight = nn.Parameter(self.conv_weight)
        self.conv.bias = nn.Parameter(self.conv_bias)
        self.conv.eval()
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x

input_tensor = torch.randn(1, 3, 10, 10)

func = Model2().to('cpu')

with torch.no_grad():
    func.train(False)
    print(func(input_tensor))
    # success
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
    # TypeError: cannot assign 'torch.FloatTensor' as parameter 'bias' (torch.nn.Parameter or None expected)

UnspecializedNNModule

class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.conv.weight = nn.Parameter(self.conv_weight)
        self.conv.bias = nn.Parameter(self.conv_bias)
        self.conv.eval()
        self.batchnorm = nn.BatchNorm2d(num_features=5)
        self.conv_weight = torch.randn(5, 3, 3, 3)
        self.conv_bias = torch.randn(5)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        return x

torch.compile

forward

__init__

NNModuleVariable

torch.compile

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):
    def forward(self, x):
        out = torch.add(x, torch.nn.Parameter(torch.ones(1)))
        return out

x = torch.randn((1, 3, 32, 32), dtype=torch.float32)

func = Model().to('cpu')

res1 = func(x)
print(res1)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # FakeTensor(..., size=(1, 3, 32, 32))

torch.compile

torch.compile(backend="eager")

Parameter

forward