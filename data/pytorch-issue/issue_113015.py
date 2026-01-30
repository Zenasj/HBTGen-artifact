import torch.nn as nn

import numpy as np
import pickle
from numpy import testing
import torch

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        mul = torch.mul(getitem, getattr_1)
        flatten = getattr_1.flatten()
        sum_1 = flatten.sum(0)
        to = flatten.to(dtype = torch.bool)
        return (mul, sum_1, to)

model_0 = Model0()
output_names_0 = ['v0_0', 'v5_0', 'v4_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        mul = torch.mul(getitem, getattr_1)
        flatten = getattr_1.flatten()
        sum_1 = flatten.sum(0)
        to = flatten.to(dtype = torch.bool)
        return (to, sum_1, mul)

model_1 = Model1()
output_names_1 = ['v11_0', 'v13_0', 'v15_0']

input_data = [v for _, v in pickle.load(open('0.pickle', 'rb')).items()]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v0_0': 'v15_0', 'v4_0': 'v11_0', 'v5_0': 'v13_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_without_complie triggers assertion")
    print(e)
print('=========================')

import numpy as np
import pickle
from numpy import testing
import torch

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        getitem_1 = getitem[(slice(-1850, -1849, 1),)]
        getitem_2 = getitem[(slice(-1850, 9223372036854775807, 43),)];  getitem = None
        neg = torch.neg(getitem_2)
        sum_1 = getitem_2.sum(0);  getitem_2 = None
        max_1 = torch.max(getitem_1, neg);  getitem_1 = neg = None
        return (sum_1, max_1)

model_0 = Model0()
output_names_0 = ['v0_0', 'v5_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        getitem_1 = getitem[(slice(-1850, -1849, 1),)]
        getitem_2 = getitem[(slice(-1850, 9223372036854775807, 43),)];  getitem = None
        neg = torch.neg(getitem_2)
        sum_1 = getitem_2.sum(0);  getitem_2 = None
        max_1 = torch.max(getitem_1, neg);  getitem_1 = neg = None
        return (max_1, sum_1)

model_1 = Model1()
output_names_1 = ['v4_0', 'v6_0']

input_data = [v for _, v in pickle.load(open('0.pickle', 'rb')).items()]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v5_0': 'v4_0', 'v0_0': 'v6_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_without_complie does not trigger assertion")
except AssertionError as e:
    print("torch_without_complie triggers assertion")
    print(e)
print('=========================')