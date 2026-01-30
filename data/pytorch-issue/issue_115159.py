import torch.nn as nn
import random

class Model0(torch.nn.Module):
    def forward(self, *args):
        mul = torch.mul(args[0], args[0])
        sin = torch.sin(mul)
        pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value = 0.5)
        return (pad, sin,)

class Model1(torch.nn.Module):
    def forward(self, *args):
        mul = torch.mul(args[0], args[0])
        transpose = mul.transpose(1, 0)
        sin = torch.sin(mul)
        pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value = 0.5)
        return (transpose, pad, sin)

import numpy as np
from numpy import testing
import torch

DEVICE='cpu'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        mul = torch.mul(args[0], args[0])
        sin = torch.sin(mul)
        pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value = 0.5)
        return (pad, sin,)

model_0 = Model0()
output_names_0 = ['v2_0', 'v4_0',]

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        mul = torch.mul(args[0], args[0])
        transpose = mul.transpose(1, 0)
        sin = torch.sin(mul)
        pad = torch.nn.functional.pad(mul, (0, 0), 'constant', value = 0.5)
        return (transpose, pad, sin)

model_1 = Model1()
output_names_1 = ['v3_0', 'v2_0', 'v4_0']

data_0 = np.random.normal(5, 1, size=(37, 30)).astype(np.float16)
input_data_0 = [data_0,]

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
output_name_dict = {'v2_0': 'v2_0', 'v4_0': 'v4_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], atol=1e-6, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
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
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], atol=1e-6, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')