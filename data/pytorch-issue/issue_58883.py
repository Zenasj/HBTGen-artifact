# torch.rand(3, 2, dtype=torch.float32)  # Input shape for the forward method

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, xind, xval):
        super().__init__()
        self.register_buffer("xind", xind)
        self.register_buffer("xval", xval)

    def forward(self, y):
        x = torch.sparse_coo_tensor(self.xind, self.xval, (2, 3))
        return torch.mm(x, y)

def my_model_function():
    i = [[0, 1, 1], [2, 0, 2]]
    v = [3, 4, 5]
    xind = torch.LongTensor(i)
    xval = torch.Tensor(v)
    return MyModel(xind, xval)

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    return torch.rand(3, 2, dtype=torch.float32)

