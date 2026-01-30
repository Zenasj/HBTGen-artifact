import torch.nn as nn

class Model0():
    def forward(self, *args):
        ceil = torch.ceil(args[0])
        sum_1 = args[1].sum(2)
        tan = torch.tan(ceil)
        clip = torch.clip(sum_1, -1.5, 1.5)
        pad = torch.nn.functional.pad(tan, (0, 0), 'constant', value = 0.5)
        add = torch.add(sum_1, pad)
        matmul = torch.matmul(pad, clip)
        return (add, matmul)

import numpy as np
import pickle
from numpy import testing
import torch

DEVICE='cuda'

class Model0(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        ceil = torch.ceil(args[0])
        sum_1 = args[1].sum(2)
        tan = torch.tan(ceil)
        clip = torch.clip(sum_1, -1.5, 1.5)
        pad = torch.nn.functional.pad(tan, (0, 0), 'constant', value = 0.5)
        add = torch.add(sum_1, pad)
        matmul = torch.matmul(pad, clip)
        return (add, matmul)

model_0 = Model0()
output_names_0 = ['v9_0', 'v8_0']

class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        ceil = torch.ceil(args[0])
        sum_1 = args[1].sum(2)
        tan = torch.tan(ceil)
        clip = torch.clip(sum_1, -1.5, 1.5)
        pad = torch.nn.functional.pad(tan, (0, 0), 'constant', value = 0.5)
        add = torch.add(sum_1, pad)
        matmul = torch.matmul(pad, clip)
        return (matmul, add)

model_1 = Model1()
output_names_1 = ['v8_0', 'v9_0']
data_0 = np.array([4.55], dtype=np.float16)
data_1 = np.array([[[5.723], [5.44 ], [6.57 ], [3.678], [6.754], [5.992], [5.477], [5.766], [4.527], [5.383], [3.697], [3.309], [5.258], [6.168], [3.02 ], [6.445], [6.54 ], [6.477], [4.707], [3.344], [4.36 ], [3.672], [3.205], [3.57 ], [5.86 ], [3.016], [5.875], [4.66 ], [6.62 ], [3.98 ], [4.754], [5.145], [6.582], [6.74 ], [4.97 ], [6.812], [4.332], [4.777], [3.346], [3.877], [3.822], [3.38 ], [4.08 ], [3.635], [5.953], [5.605], [5.258], [6.992], [4.832], [4.023], [3.613]]], dtype=np.float16)
input_data_0 = [data_0, data_1]

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
output_name_dict = {'v8_0': 'v8_0', 'v9_0': 'v9_0'}

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