import torch.nn as nn
import random

class Model0():
    def forward(self, *args):
        v0_0 = self.v0_0
        sigmoid = torch.sigmoid(v0_0)
        mul = torch.mul(args[0], sigmoid)
        pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value = 0.5)
        lt = torch.lt(pad, mul)
        return (lt)

class Model1():
    def forward(self, *args):
        v0_0 = self.v0_0
        sigmoid = torch.sigmoid(v0_0)
        mul = torch.mul(args[0], sigmoid)
        pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value = 0.5)
        lt = torch.lt(pad, mul)
        cat = torch.cat((pad,), dim = 2)
        return (lt, cat)

import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

p0 = torch.tensor(0.0002, dtype=torch.float16, device='cuda:0')

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0_0 = p0

    def forward(self, *args):
        v0_0 = self.v0_0
        sigmoid = torch.sigmoid(v0_0)
        mul = torch.mul(args[0], sigmoid)
        pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value = 0.5)
        lt = torch.lt(pad, mul)
        cat = torch.cat((pad,), dim = 2)
        return (lt, cat)

model_0 = Model0()
output_names_0 = ['v5_0', 'v0_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v0_0 = p0

    def forward(self, *args):
        v0_0 = self.v0_0
        sigmoid = torch.sigmoid(v0_0)
        mul = torch.mul(args[0], sigmoid)
        pad = torch.nn.functional.pad(mul, (-8, 0, 0, 0, 0, 0), 'constant', value = 0.5)
        lt = torch.lt(pad, mul)
        return (lt)

model_1 = Model1()
output_names_1 = ['v5_0']
data_0 = np.random.normal(5, 1, size=(1, 57, 8, 9)).astype(np.float16)
input_data_0 = [data_0]
optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

input_data_1 = input_data_0

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
model_out_1 = [v.to(DEVICE).detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.to(DEVICE).detach()]
model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v5_0': 'v5_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
model_out_0 = [v.to(DEVICE).detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.to(DEVICE).detach()]
model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
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