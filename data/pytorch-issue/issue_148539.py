import torch.nn as nn

import torch

class bad_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g * 0.5

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1, dtype=torch.float))

    def forward(self, x):
        return bad_func.apply(x * self.param)

m = Model()
t = torch.tensor([1.0, -1.0], dtype=torch.float)

def check_grad(model):
    sm = model(t).square().sum()
    print(sm)
    sm.backward()
    print(type(model), model.param.grad)
    model.param.grad = None

check_grad(m)

m_c = torch.export.export(m, (t,))
check_grad(m_c.module())