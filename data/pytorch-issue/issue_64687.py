import torch.nn as nn

import torch

class MyTensor():
    
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # ignore all errors.
        print('called!')
        pass

# torch function called.
t1 = MyTensor()
t2 = torch.nn.Parameter(torch.rand(2, 2))
torch.add(t2, t1)

# torch function not called.
inp = torch.rand(10, 10)
torch.nn.functional.linear(inp, t1, t2)