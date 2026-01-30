import torch.nn as nn

class Model0():
    def forward(self, *args):
        _args = args
        v11_0 = self.v11_0
        getitem = _args[0]
        abs_1 = torch.abs(v11_0)
        div = torch.div(getitem, abs_1)
        cat = torch.cat((div,), dim = 2)
        transpose = div.transpose(2, 4)
        cat_1 = torch.cat((transpose,), dim = 4)
        floor = torch.floor(cat_1)
        floor_1 = torch.floor(floor)
        mean = floor_1.mean(4)
        interpolate = torch.nn.functional.interpolate(floor_1, size = [1, 2, 10], scale_factor = None, mode = 'trilinear', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (cat, mean, interpolate)

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE='cuda'

p0 = torch.nn.Parameter(torch.empty([57, 1, 10, 1, 40], dtype=torch.float16), requires_grad=False).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v11_0 = p0

    def forward(self, *args):
        _args = args
        v11_0 = self.v11_0
        getitem = _args[0]
        abs_1 = torch.abs(v11_0)
        div = torch.div(getitem, abs_1)
        cat = torch.cat((div,), dim = 2)
        transpose = div.transpose(2, 4)
        cat_1 = torch.cat((transpose,), dim = 4)
        floor = torch.floor(cat_1)
        floor_1 = torch.floor(floor)
        mean = floor_1.mean(4)
        interpolate = torch.nn.functional.interpolate(floor_1, size = [1, 2, 10], scale_factor = None, mode = 'trilinear', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (cat, mean, interpolate)

model_0 = Model0()
output_names_0 = ['v2_0', 'v0_0', 'v7_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v11_0 = p0

    def forward(self, *args):
        _args = args
        v11_0 = self.v11_0
        getitem = _args[0]
        abs_1 = torch.abs(v11_0)
        div = torch.div(getitem, abs_1)
        cat = torch.cat((div,), dim = 2)
        transpose = div.transpose(2, 4)
        cat_1 = torch.cat((transpose,), dim = 4)
        floor = torch.floor(cat_1)
        floor_1 = torch.floor(floor)
        mean = floor_1.mean(4)
        interpolate = torch.nn.functional.interpolate(floor_1, size = [1, 2, 10], scale_factor = None, mode = 'trilinear', align_corners = None, recompute_scale_factor = None, antialias = False)
        return (interpolate, cat, mean)

model_1 = Model1()
output_names_1 = ['v7_0', 'v2_0', 'v0_0']

data_0 = np.array(3.273, dtype=np.float16)
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
output_name_dict = {'v7_0': 'v7_0', 'v2_0': 'v2_0', 'v0_0': 'v0_0'}

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