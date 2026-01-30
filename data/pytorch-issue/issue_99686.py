import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.split(x, [3, 2, 3], dim=1)
        x = torch.cat([x[1], x[0], x[2]], dim=1)
        return x

input_tensor = torch.randn(1, 8)

func = Model().to('cpu')

print(input_tensor)
# tensor([[-1.6977,  0.6374,  0.0781, -0.4140,  1.5172,  0.0473,  0.8435, -0.2261]])

res1 = func(input_tensor)
print(res1)
# tensor([[-0.4140,  1.5172, -1.6977,  0.6374,  0.0781,  0.0473,  0.8435, -0.2261]])

jit_func = torch.compile(func)
res2 = jit_func(input_tensor)
print(res2)
# tensor([[-1.6977,  0.6374,  0.0781, -0.4140,  1.5172,  0.0473,  0.8435, -0.2261]])

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input):
        split_node = torch.split(input, [2, 1, 1], dim=1)
        cat_node = torch.cat([split_node[0], split_node[1], split_node[2]], dim=1)
        return cat_node

input_tensor = torch.randn(1, 5)
print(input_tensor)
# tensor([[-1.6977,  0.6374,  0.0781, -0.4140,  1.5172]])

func = Model().to('cpu')

jit_func = torch.compile(func)
res2 = jit_func(input_tensor)
print(res2)
# tensor([[-1.6977,  0.6374,  0.0781, -0.4140,  1.5172]])

res1 = func(input_tensor)
print(res1)
# RuntimeError: split_with_sizes expects split_sizes to sum exactly to 5 (input tensor's size at dimension 1), but got split_sizes=[2, 1, 1]