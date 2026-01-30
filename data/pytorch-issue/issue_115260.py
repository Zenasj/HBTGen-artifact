import torch.nn as nn
import random

class Model0(torch.nn.Module):
    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 2)
        max_1 = torch.max(args[4], p0)
        mul = torch.mul(cat, max_1)
        tan = torch.tan(mul)
        return (tan)

import numpy as np
from numpy import testing
import torch

DEVICE='cpu'

p0 = torch.tensor([1.0879], dtype=torch.float16).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 2)
        max_1 = torch.max(args[4], p0)
        mul = torch.mul(cat, max_1)
        tan = torch.tan(mul)
        return (tan,)

model_0 = Model0()
output_names_0 = ['v1_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((args[3], args[2], args[1], args[0]), dim = 2)
        max_1 = torch.max(args[4], p0)
        mul = torch.mul(cat, max_1)
        tan = torch.tan(mul)
        return (mul, tan)

model_1 = Model1()
output_names_1 = ['v15_0', 'v1_0']

data_0 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
data_1 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
data_2 = np.random.normal(5, 1, size=(17, 5, 11, 7)).astype(np.float16)
data_3 = np.random.normal(5, 1, size=(17, 5, 1, 7)).astype(np.float16)
data_4 = np.array(4.39, dtype=np.float16)
input_data_0 = [data_0,data_1,data_2,data_3,data_4,]

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