import torch.nn as nn

import numpy as np
from numpy import testing
import torch

DEVICE='cpu'

p0 = torch.tensor(5, dtype=torch.int8).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v8_0 = p0

    def forward(self, *args):
        _args = args
        v8_0 = self.v8_0
        getitem = _args[0]
        getitem_1 = _args[1]
        add = torch.add(getitem_1, v8_0)
        mul = torch.mul(add, v8_0)
        mul_1 = torch.mul(getitem, mul)
        abs_1 = torch.abs(mul_1)
        return (abs_1)

model_0 = Model0()
output_names_0 = ['v1_0']

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
        mul_2 = torch.mul(v0_0, getitem)
        mul_3 = torch.mul(v0_0, mul)
        mul_4 = torch.mul(getitem_1, mul_2)
        add_3 = torch.add(mul_4, mul_3)
        abs_1 = torch.abs(add_3)
        return (abs_1)

model_1 = Model1()
output_names_1 = ['v17_0']

input_data_0 = np.array(5, dtype=np.int8)
input_data_1 = np.array(4, dtype=np.int8)
input_data = [input_data_0, input_data_1]

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
output_name_dict = {'v1_0': 'v17_0'}

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