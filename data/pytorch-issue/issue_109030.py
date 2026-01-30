import torch.nn as nn

import numpy as np
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v9_0 = torch.nn.Parameter(torch.tensor([4], dtype=torch.int8), requires_grad=False)
        self.v8_0 = torch.nn.Parameter(torch.tensor([5], dtype=torch.int8), requires_grad=False)
        self.v6_0 = torch.nn.Parameter(torch.tensor([6], dtype=torch.int8), requires_grad=False)


    def forward(self, *args):
        _args = args
        getitem = _args[0]
        getitem_1 = _args[1]
        getitem_2 = _args[2];  _args = None
        flatten = getitem.flatten();  getitem = None
        v6_0 = self.v6_0
        v8_0 = self.v8_0
        v9_0 = self.v9_0
        cat = torch.cat((getitem_1, v6_0, flatten, v8_0, v9_0), dim = 0);  getitem_1 = v6_0 = flatten = v8_0 = v9_0 = None
        sum_1 = cat.sum(0)
        sub = torch.sub(cat, getitem_2);  cat = getitem_2 = None
        return (sum_1, sub)

m = M()

inp = [np.array([[[[4]]]], dtype='int8'), np.array([3,6,5,7,3,6,3,4,7,5,6,4,4,4,4,5,7,4,7,4,7,6,5,7,6,6,3,4,6,7,5,5,5,4], dtype='int8'), np.array(7, dtype='int8')]

opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)

m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp])
print(f"torch result: {m_out}")
opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp])
print(f"torch.compile result: {opt_out}")

import numpy as np
import torch
import pickle

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.tensor([6, 7, 4, 4, 5, 4, 6, 6, 7, 3, 4, 5, 3, 4, 5, 5, 7, 6, 4, 4, 7, 6, 6, 4, 4, 6, 7, 3, 7, 3, 6, 5, 6, 5, 5, 5, 5, 4, 6, 4, 7, 7, 4, 6, 5, 4, 4, 5, 4, 7, 7], dtype=torch.int16), requires_grad=False)
    def forward(self, *args):
        _args = args
        getitem = _args[0];  _args = None

        cat = torch.cat((self.x, getitem), dim = 0);  _tensor_constant0 = getitem = None
        sum_1 = cat.sum(0)
        return sum_1

m = M()

inp = [np.array([3], dtype='int16')]

opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)

m_out = m(*[torch.from_numpy(v).to('cpu') for v in inp]).cpu().detach()

opt_out = opt(*[torch.from_numpy(v).to('cpu') for v in inp]).cpu().detach()

print(m_out)
print(opt_out)