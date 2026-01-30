import torch.nn as nn

import numpy as np
import pickle
from numpy import testing
import torch

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        pad = torch.nn.functional.pad(getitem, (46, 0), 'constant', value = 0.5)
        mean = pad.mean(0)
        to = pad.to(dtype = torch.bool)
        gt = torch.gt(mean, pad)
        return (to, gt)

model_0 = Model0()
output_names_0 = ['v4_0', 'v3_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        pad = torch.nn.functional.pad(getitem, (46, 0), 'constant', value = 0.5)
        mean = pad.mean(0)
        to = pad.to(dtype = torch.bool)
        gt = torch.gt(mean, pad)
        return (gt, to)

model_1 = Model1()
output_names_1 = ['v3_0', 'v2_0']

input_data_0 = np.array([[6.46 ], [3.229], [5.785], [5.145], [6.676], [3.998], [5.77 ], [3.09 ], [4.66 ], [6.348], [4.273], [5.133], [3.367], [5.7  ], [3.979], [5.836], [4.543], [6.586], [3.504], [3.416], [3.117], [4.58 ], [5.793], [5.24 ], [5.566], [6.688], [3.459], [6.13 ], [3.07 ], [6.824], [4.91 ], [6.938], [4.38 ], [3.69 ], [5.324], [4.957], [4.12 ], [3.271], [5.375], [4.223], [3.71 ], [3.252], [6.504], [3.713], [5.285], [4.145], [3.746], [5.414], [3.84 ], [6.08 ], [6.457], [3.57 ], [5.805], [3.318], [4.215], [4.473]], dtype=np.float16)
input_data = [input_data_0]

optmodel_0 = torch.compile(model_0, fullgraph=True, backend='inductor', mode=None)
model_out_0 = optmodel_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

optmodel_1 = torch.compile(model_1, fullgraph=True, backend='inductor', mode=None)
model_out_1 = optmodel_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
output_1 = dict(zip(output_names_1, model_out_1))
output_name_dict = {'v4_0': 'v2_0', 'v3_0': 'v3_0'}

print('=========================')
try:
    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(output_0[tensor_name_0], output_1[tensor_name_1], rtol=1)
    print("torch_complie does not trigger assertion")
except AssertionError as e:
    print("torch_complie triggers assertion")
    print(e)
print('=========================')

model_out_0 = model_0(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_0 = [v.cpu().detach() for v in model_out_0] if isinstance(model_out_0, tuple) else [model_out_0.cpu().detach()]
model_out_0 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_0]
output_0 = dict(zip(output_names_0, model_out_0))

model_out_1 = model_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
model_out_1 = [v.cpu().detach() for v in model_out_1] if isinstance(model_out_1, tuple) else [model_out_1.cpu().detach()]
model_out_1 = [v.resolve_conj().numpy() if v.is_conj() else v.numpy() for v in model_out_1]
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

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE="cuda"

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        pad = torch.nn.functional.pad(getitem, (46, 0), 'constant', value = 0.5)
        mean = pad.mean(0)
        to = pad.to(dtype = torch.bool)
        gt = torch.gt(mean, pad)
        return (to, gt)

model_0 = Model0()
output_names_0 = ['v4_0', 'v3_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None
        pad = torch.nn.functional.pad(getitem, (46, 0), 'constant', value = 0.5)
        mean = pad.mean(0)
        to = pad.to(dtype = torch.bool)
        gt = torch.gt(mean, pad)
        return (gt, to)

model_1 = Model1()
output_names_1 = ['v3_0', 'v2_0']

input_data_0 = np.array([[6.46 ], [3.229], [5.785], [5.145], [6.676], [3.998], [5.77 ], [3.09 ], [4.66 ], [6.348], [4.273], [5.133], [3.367], [5.7  ], [3.979], [5.836], [4.543], [6.586], [3.504], [3.416], [3.117], [4.58 ], [5.793], [5.24 ], [5.566], [6.688], [3.459], [6.13 ], [3.07 ], [6.824], [4.91 ], [6.938], [4.38 ], [3.69 ], [5.324], [4.957], [4.12 ], [3.271], [5.375], [4.223], [3.71 ], [3.252], [6.504], [3.713], [5.285], [4.145], [3.746], [5.414], [3.84 ], [6.08 ], [6.457], [3.57 ], [5.805], [3.318], [4.215], [4.473]], dtype=np.float16)
input_data = [input_data_0]

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
output_name_dict = {'v4_0': 'v2_0', 'v3_0': 'v3_0'}

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

model_out_1 = model_1(*[torch.from_numpy(v).to('cpu') for v in input_data])
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

def test_to_bool_and_gt(self):
        # https://github.com/pytorch/pytorch/issues/113014
        def fn(x):
            pad = torch.nn.functional.pad(x, (46, 0), 'constant', value=0.5)
            mean = pad.mean(0)
            to = pad.to(torch.bool)
            gt = torch.gt(mean, pad)
            return (to, gt)

        x = torch.randn((56, 1))
        self.common(fn, (x,))