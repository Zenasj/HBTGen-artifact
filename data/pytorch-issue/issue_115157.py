import torch.nn as nn
import random

class Model0(torch.nn.Module):
    def forward(self, *args):
        m1 = m0(args[0])
        abs_1 = torch.abs(m1)
        pad = torch.nn.functional.pad(abs_1, (0, 24, 34, 0), 'replicate')
        add = torch.add(pad, pad)
        return (add,)

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE='cpu'

m0 = torch.nn.MaxPool2d(kernel_size=(1, 1), stride=1, padding=0, dilation=1, ceil_mode=False)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        m1 = m0(args[0])
        abs_1 = torch.abs(m1)
        pad = torch.nn.functional.pad(abs_1, (0, 24, 34, 0), 'replicate')
        add = torch.add(pad, pad)
        return (add,)

model_0 = Model0()
output_names_0 = ['v1_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        m1 = m0(args[0])
        abs_1 = torch.abs(m1)
        pad = torch.nn.functional.pad(abs_1, (0, 24, 34, 0), 'replicate')
        add = torch.add(pad, pad)
        return (pad, add)

model_1 = Model1()
output_names_1 = ['v8_0', 'v1_0']

data_0 = np.random.normal(5, 1, size=(1, 45, 1, 1)).astype(np.float32)
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
output_name_dict = {'v1_0': 'v1_0'}

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