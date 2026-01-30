import torch.nn as nn

py
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(420)

class M(torch.nn.Module):
    def forward(self, input_data):
        output = torch.nn.functional.hardshrink((torch.sign(input_data) * input_data), 0.7)
        return output
func = M().to('cpu')

x = torch.tensor(np.array([(- 0.9), 0.9]))


with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)
    # tensor([-0.9000,  0.9000], dtype=torch.float64)

    res2 = jit_func(x)
    print(res2)
    # tensor([0., 0.], dtype=torch.float64)

    torch.testing.assert_close(res1, res2, rtol=1e-3, atol=1e-3)

py
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + 3
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 * torch.div(1, 6)
        return v5

func = Model()

x = torch.randn(1, 3)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)
    # tensor([[0.5911]], device='cuda:0')

    res2 = jit_func(x)
    print(res2)
    # tensor([[0.]], device='cuda:0')

    torch.testing.assert_close(res1, res2, rtol=1e-3, atol=1e-3)