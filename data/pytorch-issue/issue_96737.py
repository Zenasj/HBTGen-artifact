___odict_getitem(np_to_torch, np.float16)

import torch
import numpy as np
from typing import Union, List

np_to_torch = {np.float16: torch.float16, np.float32: torch.float32, bool: torch.bool}


class Device:
    def __init__(self, device, device_type):
        self.dev = device
        self.device_type = device_type
        
    def create_tensor(self, shape, dtype):
        dtype = np_to_torch[dtype]
        data = torch.empty(shape, dtype=dtype, device=self.dev)
        return data

class Model:
    def __init__(self, device):
        self.device = device
        
    def forward(self,
                 inputs: Union[np.array, List[List[int]]]):
        x = self.device.create_tensor(inputs, np.float16)
        y = self.device.create_tensor(inputs, np.float16)
        return x + y
    
dev = Device(torch.device('cuda:0'), 'GPU')
model = Model(dev)

def wrapper_fn(inputs: Union[np.array, List[List[int]]]):
    return model.forward(inputs)

generate = torch.compile(wrapper_fn)

generate((2, 3, 4))

def as_python_constant(self):
        return self.value

import torch
d = {int: torch.float}
@torch.compile
def f(x, y):
    return torch.randn(5, dtype=d[y])

f(torch.zeros(4), int)