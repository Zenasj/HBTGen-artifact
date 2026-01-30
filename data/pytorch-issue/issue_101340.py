import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        t1 = torch.tanh(x)
        t2 = torch.sign(t1)
        t3 = torch.min(t1, t2)
        return t3

func = Model().to('cpu')

x = torch.randn(1, 3)

with torch.no_grad():
    func = func.eval()
    res1 = func(x) # without jit
    print(res1)

    jit_func = torch.compile(func)
    res2 = jit_func(x)
    # torch._inductor.exc.CppCompileError: C++ compile error
    # no matching function for call to ‘min(float&, int&)’