import torch.nn as nn

import numpy as np
from numpy import testing
import torch

DEVICE='cpu'

p0 = torch.tensor(6, dtype=torch.int8).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v7_0 = p0

    def forward(self, *args):
        _args = args
        v7_0 = self.v7_0
        getitem = _args[0]
        getitem_1 = _args[1]
        mul = torch.mul(getitem, v7_0)
        mul_1 = torch.mul(mul, getitem_1)
        add = torch.add(mul_1, mul_1)
        to = add.to(dtype = torch.int32)
        return (to,)

model_0 = Model0()
output_names_0 = ['v5_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0_0 = p0

    def forward(self, *args):
        _args = args
        v0_0 = self.v0_0
        getitem = _args[0]
        getitem_1 = _args[1]
        add_1 = torch.add(v0_0, v0_0)
        mul_1 = torch.mul(getitem, add_1)
        mul_4 = torch.mul(mul_1, getitem_1)
        to = mul_4.to(dtype = torch.int32)
        return (to)

model_1 = Model1()
output_names_1 = ['v12_0']

input_data_0 = np.array(4, dtype=np.int8)
input_data_1 = np.array([4, 5, 7, 3, 6, 6, 4, 5, 5, 6, 7, 5, 5, 4], dtype=np.int8)
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
output_name_dict = {'v5_0': 'v12_0'}

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