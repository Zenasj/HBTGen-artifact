import torch.nn as nn

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE='cpu'

p0 = torch.nn.Parameter(torch.empty([40, 30], dtype=torch.uint8), requires_grad=False).to(DEVICE)
p1 = torch.nn.Parameter(torch.empty([30], dtype=torch.uint8), requires_grad=False).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v4_0 = p0
        self.v1_0 = p1

    def forward(self, *args):
        _args = args
        v4_0 = self.v4_0
        v1_0 = self.v1_0
        getitem = _args[0]
        reshape = v1_0.reshape(30)
        neg = torch.neg(v4_0)
        add = torch.add(reshape, getitem)
        mul = torch.mul(add, neg)
        sum_1 = mul.sum(0)
        return (sum_1,)

model_0 = Model0()
output_names_0 = ['v0_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v4_0 = p0
        self.v1_0 = p1

    def forward(self, *args):
        _args = args
        v4_0 = self.v4_0
        v1_0 = self.v1_0
        getitem = _args[0]
        reshape = v1_0.reshape(30)
        neg = torch.neg(v4_0)
        add = torch.add(reshape, getitem)
        mul = torch.mul(neg, add)
        sum_1 = mul.sum(0)
        return (sum_1)

model_1 = Model1()
output_names_1 = ['v9_0']

input_data = [v for _, v in pickle.load(open('0.pickle', 'rb')).items()]

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
output_name_dict = {'v0_0': 'v9_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
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
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')