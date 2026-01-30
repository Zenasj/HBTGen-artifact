import torch.nn as nn

class Model0():
    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        sin = torch.sin(getitem)
        gelu = torch._C._nn.gelu(getitem)
        sigmoid = torch.sigmoid(gelu)
        sub = torch.sub(gelu, gelu)
        return (sin, sigmoid, sub)

import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        sin = torch.sin(getitem)
        gelu = torch._C._nn.gelu(getitem)
        sigmoid = torch.sigmoid(gelu)
        sub = torch.sub(gelu, gelu)
        return (sin, sigmoid, sub)

model_0 = Model0()
output_names_0 = ['v5_0', 'v4_0', 'v1_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        sin = torch.sin(getitem)
        gelu = torch._C._nn.gelu(getitem)
        sigmoid = torch.sigmoid(gelu)
        sub = torch.sub(gelu, gelu)
        return (sigmoid, sub, sin)

model_1 = Model1()
output_names_1 = ['v4_0', 'v1_0', 'v5_0']

input_data = [np.array([5.2382936, 3.8000417, 6.932271 , 3.3283236, 6.4508905, 3.1469257,
       3.5212083, 3.0663643, 3.6943512, 3.4826405, 4.1576777, 4.895685 ,
       6.930315 , 4.158941 , 4.80448  , 6.1907043, 4.3918133, 4.152346 ,
       6.03102  , 5.4802165, 3.8566995, 4.251566 , 6.468913 , 3.3474746,
       5.7752266], dtype=np.float32)]

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
output_name_dict = {'v4_0': 'v4_0', 'v5_0': 'v5_0', 'v1_0': 'v1_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
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
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}')
    print("torch_eager does not trigger assertion")
except AssertionError as e:
    print("torch_eager triggers assertion")
    print(e)
print('=========================')