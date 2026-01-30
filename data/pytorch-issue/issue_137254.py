import torch.nn as nn

import torch

class Model(torch.nn.Module):

    def forward(self, x):
        s1 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        s2 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        return torch.cat((s1, s2), -1)  # If dim=0, the res is OK!


x = torch.randn(2, 2)
test_inputs = [x]

func = Model().to('cuda')  # If I use CPU, the result is same
compiled_func = torch.compile(func, backend="inductor")  # backend="cudagraphs" is OK

print("-" * 10 + "original model" + "-" * 10)
print(func(*test_inputs))

print("-" * 10 + "compiled model" + "-" * 10)
print(compiled_func(*test_inputs))

import torch

class Model(torch.nn.Module):

    def forward(self, x):
        s1 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        s2 = torch.addmm(x, x, x, beta=0.0, alpha=0.0)
        print(s1, '\n', s2)
        return torch.cat((s1, s2), -1)  # If dim=0, the res is OK!


x = torch.randn(2, 2)
test_inputs = [x]

func = Model().to('cuda')  # If I use CPU, the result is same
compiled_func = torch.compile(func, backend="inductor")  # backend="cudagraphs" is OK

print("-" * 10 + "original model" + "-" * 10)
print(func(*test_inputs))

print("-" * 10 + "compiled model" + "-" * 10)
print(compiled_func(*test_inputs))