import torch.nn as nn
import random

class Model0():
    def forward(self, *args):
        cat = torch.cat((p2, args[0], p1, p0), dim = 0)
        mul = torch.mul(args[1], args[2])
        squeeze = mul.squeeze(0)
        max_1 = torch.max(squeeze, cat)
        cat_2 = torch.cat((max_1,), dim = 1)
        sin = torch.sin(cat_2)
        return (sin,)

class Model1():
    def forward(self, *args):
        cat = torch.cat((p2, args[0], p1, p0), dim = 0)
        mul = torch.mul(args[1], args[2])
        cat_1 = torch.cat((mul, mul), dim = 1)
        squeeze = mul.squeeze(0)
        max_1 = torch.max(squeeze, cat)
        cat_2 = torch.cat((max_1,), dim = 1)
        sin = torch.sin(cat_2)
        return (cat_1, sin)

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE='cuda'

p0_np = np.random.normal(0, 1, size=(1, 61)).astype(np.float16)
p0 = torch.from_numpy(p0_np).to(DEVICE)
p1_np = np.random.normal(0, 1, size=(54, 61)).astype(np.float16)
p1 = torch.from_numpy(p1_np).to(DEVICE)
p2_np = np.random.normal(0, 1, size=(1, 61)).astype(np.float16)
p2 = torch.from_numpy(p2_np).to(DEVICE)

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((p2, args[0], p1, p0), dim = 0)
        mul = torch.mul(args[1], args[2])
        squeeze = mul.squeeze(0)
        max_1 = torch.max(squeeze, cat)
        cat_2 = torch.cat((max_1,), dim = 1)
        sin = torch.sin(cat_2)
        return (sin,)

model_0 = Model0()
output_names_0 = ['v3_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        cat = torch.cat((p2, args[0], p1, p0), dim = 0)
        mul = torch.mul(args[1], args[2])
        cat_1 = torch.cat((mul, mul), dim = 1)
        squeeze = mul.squeeze(0)
        max_1 = torch.max(squeeze, cat)
        cat_2 = torch.cat((max_1,), dim = 1)
        sin = torch.sin(cat_2)
        return (cat_1, sin)

model_1 = Model1()
output_names_1 = ['v2_0', 'v3_0']

data_0 = np.array([[3.123, 4.883, 3.256, 6.31 , 5.766, 4.004, 3.031, 4.016, 5.945,
        5.1  , 5.305, 3.031, 5.566, 4.08 , 5.707, 6.63 , 3.139, 6.043,
        6.953, 6.26 , 3.666, 6.56 , 5.1  , 3.205, 3.3  , 3.006, 4.758,
        3.412, 3.268, 3.031, 6.215, 3.588, 3.697, 4.035, 6.008, 5.207,
        6.05 , 3.844, 6.434, 4.863, 6.27 , 6.438, 3.385, 3.385, 5.21 ,
        5.05 , 6.766, 3.033, 3.33 , 3.598, 3.475, 6.43 , 4.598, 5.96 ,
        6.906, 6.875, 3.637, 5.562, 3.86 , 3.648, 3.227]], dtype=np.float16)
data_1 = np.array([[[6.188]]], dtype=np.float16)
data_2 = np.array([[[3.55 , 3.547, 6.867, 4.652, 6.24 , 3.61 , 5.727, 5.566, 6.984,
         3.084, 5.48 , 5.17 , 5.293, 3.623, 6.6  , 6.008, 6.867, 6.074,
         4.473, 3.305, 5.45 , 3.098, 6.3  , 3.945, 6.824, 6.875, 3.004,
         3.346, 5.203, 6.684, 3.209, 6.625, 6.07 , 4.957, 3.512, 6.945,
         5.258, 3.268, 5.965, 3.244, 6.07 , 3.424, 5.945, 6.914, 5.344,
         6.22 , 7.   , 5.48 , 4.58 , 3.209, 5.367, 3.115, 4.66 , 5.35 ,
         5.54 , 4.164, 6.64 , 4.55 , 5.395, 4.27 , 6.   ]]], dtype=np.float16)
input_data_0 = [data_0,data_1,data_2,]

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
output_name_dict = {'v3_0': 'v3_0'}

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