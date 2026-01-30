import torch.nn as nn

import numpy as np
import pickle
from numpy import testing
import torch

p0 = torch.tensor(6)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v6_0 = p0

    def forward(self, *args):
        _args = args
        v6_0 = self.v6_0
        getitem = _args[0]
        getitem_1 = _args[1];  _args = None
        add = torch.add(getitem, v6_0);  getitem = v6_0 = None
        cat = torch.cat((getitem_1,), dim = 0)
        sub = torch.sub(getitem_1, add)
        matmul = torch.matmul(sub, sub)
        max_1 = sub.max(0)
        getattr_1 = max_1.values
        return (cat, matmul, getattr_1)

model_0 = Model0()
output_names_0 = ['v4_0', 'v2_0', 'v1_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v6_0 = p0

    def forward(self, *args):
        _args = args
        v6_0 = self.v6_0
        getitem = _args[0]
        getitem_1 = _args[1];  _args = None
        add = torch.add(getitem, v6_0);  getitem = v6_0 = None
        cat = torch.cat((getitem_1,), dim = 0)
        sub = torch.sub(getitem_1, add)
        matmul = torch.matmul(sub, sub)
        max_1 = sub.max(0)
        getattr_1 = max_1.values
        return (matmul, getattr_1, cat)

model_1 = Model1()
output_names_1 = ['v5_0', 'v7_0', 'v9_0']

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
output_name_dict = {'v2_0': 'v5_0', 'v4_0': 'v9_0', 'v1_0': 'v7_0'}

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