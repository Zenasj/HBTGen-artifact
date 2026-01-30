import torch.nn as nn
import random

class Model0():
    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 0)
        pad = torch.nn.functional.pad(cat, (0, 0, 30, 0), 'replicate')
        cos = torch.cos(pad)
        floor = torch.floor(cos)
        div = torch.div(cos, floor)
        pad_1 = torch.nn.functional.pad(cos, (0, 0, 0, 0, 0, 0), 'replicate')
        interpolate = torch.nn.functional.interpolate(floor, size = [1, 1], scale_factor = None, mode = 'nearest', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (pad_1, interpolate, div)

import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 0)
        pad = torch.nn.functional.pad(cat, (0, 0, 30, 0), 'replicate')
        cos = torch.cos(pad)
        floor = torch.floor(cos)
        div = torch.div(cos, floor)
        pad_1 = torch.nn.functional.pad(cos, (0, 0, 0, 0, 0, 0), 'replicate')
        interpolate = torch.nn.functional.interpolate(floor, size = [1, 1], scale_factor = None, mode = 'nearest', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (pad_1, interpolate, div)

model_0 = Model0()
output_names_0 = ['v8_0', 'v6_0', 'v5_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 0)
        pad = torch.nn.functional.pad(cat, (0, 0, 30, 0), 'replicate')
        cos = torch.cos(pad)
        floor = torch.floor(cos)
        div = torch.div(cos, floor)
        pad_1 = torch.nn.functional.pad(cos, (0, 0, 0, 0, 0, 0), 'replicate')
        interpolate = torch.nn.functional.interpolate(floor, size = [1, 1], scale_factor = None, mode = 'nearest', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (div, pad_1, interpolate)

model_1 = Model1()
output_names_1 = ['v5_0', 'v8_0', 'v6_0']

data_0 = np.random.normal(5, 1, size=(25, 53, 1, 1)).astype(np.float16)
data_1 = np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)
data_2 = np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)
data_3 = np.random.normal(5, 1, size=(1, 53, 1, 1)).astype(np.float16)
input_data_0 = [data_0,data_1,data_2,data_3,]

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
output_name_dict = {'v5_0': 'v5_0', 'v8_0': 'v8_0', 'v6_0': 'v6_0'}

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