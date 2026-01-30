import torch.nn as nn

import numpy as np
import pickle
from numpy import testing
import torch

p0 = torch.nn.Conv2d(39, 1, kernel_size=(1, 17), stride=(2, 2))

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m4 = p0

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        max_1 = getitem.max(3)
        getattr_1 = max_1.values
        relu = torch.relu(getitem)
        m4 = self.m4(getattr_1)
        return (relu, m4)

model_0 = Model0()
output_names_0 = ['v0_0', 'v5_0']

model_1 = Model0()
output_names_1 = ['v2_0', 'v5_0']

input_data = [v for _, v in pickle.load(open('0.pickle', 'rb')).items()]

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
output_name_dict = {'v0_0':'v2_0', 'v5_0': 'v5_0'}

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
    print("torch_without_complie does not trigger assertion")
except AssertionError as e:
    print("torch_without_complie triggers assertion")
    print(e)
print('=========================')