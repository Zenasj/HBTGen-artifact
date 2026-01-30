import torch.nn as nn
import random

class Model0():
    def __init__(self):
        self.v0_0 = p0
    def forward(self, *args):
        _args = args
        v0_0 = self.v0_0
        getitem = _args[0]
        getitem_1 = _args[1]
        mul = torch.mul(v0_0, getitem)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (gt)

class Model1():
    def __init__(self):
        self.v0_0 = p0
    def forward(self, *args):
        _args = args
        v0_0 = self.v0_0
        getitem = _args[0]
        getitem_1 = _args[1]
        mul = torch.mul(v0_0, getitem)
        cat_1 = torch.cat((mul, mul), dim = 0)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (cat_1, gt)

import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

p0 = torch.tensor([5.6484], device='cuda:0', dtype=torch.float16)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0_0 = p0

    def forward(self, *args):
        _args = args
        v0_0 = self.v0_0
        getitem = _args[0]
        getitem_1 = _args[1]
        mul = torch.mul(v0_0, getitem)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (gt)

model_0 = Model0()
output_names_0 = ['v7_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0_0 = p0

    def forward(self, *args):
        _args = args
        v0_0 = self.v0_0
        getitem = _args[0]
        getitem_1 = _args[1]
        mul = torch.mul(v0_0, getitem)
        cat_1 = torch.cat((mul, mul), dim = 0)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (cat_1, gt)

model_1 = Model1()
output_names_1 = ['v5_0','v7_0']

input_data = [np.random.rand(1,21,1,40).astype(np.float16),
              np.random.rand(1,1,21,18,1).astype(np.float16)]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data])
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v7_0': 'v7_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data])
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')