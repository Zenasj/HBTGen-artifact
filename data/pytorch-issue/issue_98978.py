py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)

    def forward(self, x):
        x = self.fc1(x.permute(1, 2, 0))
        return x
input_tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

func = Net().to('cpu')

with torch.no_grad():
    jit_func = torch.compile(func)
    print(jit_func(input_tensor))
    #tensor([[[  6.8708, -10.1139,  -2.9715],
    #     [  7.4253, -11.1465,  -2.5976],
    #     [  7.9799, -12.1791,  -2.2237]],
    #    [[  8.5344, -13.2118,  -1.8498],
    #     [  9.0889, -14.2444,  -1.4759],
    #     [  9.6435, -15.2770,  -1.1020]],
    #    [[ 10.1980, -16.3097,  -0.7281],
    #     [ 10.7526, -17.3423,  -0.3542],
    #     [ 11.3071, -18.3750,   0.0197]]])
    print(func(input_tensor))
    # RuntimeError: expected scalar type Long but found Float

py
import torch
import torch.nn as nn

torch.manual_seed(420)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)

    def forward(self, x):
        x = self.fc1(x.permute(1, 2, 0))
        return x
input_tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

func = Net().to('cpu')

jit_func = torch.compile(func)
print(jit_func(input_tensor))
# torch._dynamo.exc.TorchRuntimeError

aten.mm

import torch
import torch.nn as nn

def func(x):
    return nn.Linear(3, 3)(x)

t = torch.arange(27).view(3, 3, 3)
out = torch.compile(func)(t)
print(out)
# Generates a 3 x 3 x 3 tensor of floats
print(func(t))
# RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Float

linear = torch._C._nn.linear(l_x_, l_stack0_weight, l_stack0_bias)

def foo(x, y):
  return torch.mm(x, y)

t1 = torch.arange(6, dtype=torch.float).view(2, 3)
t2 = torch.arange(9, dtype=torch.int64).view(3, 3)
foo(t1, t2)