import torch
class MyFloatTensor(torch.FloatTensor):
    pass
MyFloatTensor()

import torch
class MyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
MyTensor(2)

import torch
class MyFloatTensor(torch.FloatTensor):
    pass
MyFloatTensor()

import torch
class MyTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
MyTensor(2)