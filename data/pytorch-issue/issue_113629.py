import torch.nn as nn

class Model0():
    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1]
        div = torch.div(getitem_1, getitem)
        sum_1 = div.sum(1)
        sub = torch.sub(div, sum_1)
        return (sub)

class Model1():
    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1]
        div = torch.div(getitem_1, getitem)
        sum_1 = div.sum(1)
        cat = torch.cat((sum_1, sum_1, sum_1), dim = 0)
        sub = torch.sub(div, sum_1)
        return (cat, sub)

import numpy as np
from numpy import testing
import torch

DEVICE='cuda'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1]
        div = torch.div(getitem_1, getitem)
        sum_1 = div.sum(1)
        cat = torch.cat((sum_1, sum_1, sum_1), dim = 0)
        sub = torch.sub(div, sum_1)
        return (cat, sub)

model_0 = Model0()
output_names_0 = ['v2_0', 'v0_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1]
        div = torch.div(getitem_1, getitem)
        sum_1 = div.sum(1)
        sub = torch.sub(div, sum_1)
        return (sub)

model_1 = Model1()
output_names_1 = ['v0_0']

input_data = [np.array(6.973, dtype=np.float16),
              np.array([[6.81 ], [6.49 ], [6.277], [4.77 ], [6.793], [5.598], [5.676], [5.008], [6.67 ], [4.08 ], [6.465], [3.543], [4.902], [3.346], [5.875], [4.414], [3.791], [5.438], [6.125], [6.08 ]], dtype=np.float16)]

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
output_name_dict = {'v0_0': 'v0_0'}

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