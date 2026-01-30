import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):
    def forward(self, x, y, inp):
        return torch.add(torch.mm(x, y), inp)

x = torch.randn(3, 4).cuda()

y = torch.randn(4, 5).cuda()

inp = torch.randn(3, 5).cuda()

func = Model()

res1 = func(x, y, inp)
print(res1)

jit_func = torch.compile(func)
res2 = jit_func(x, y, inp)
print(res2)
# TypeError: can't multiply sequence by non-int of type 'float'
# While executing %mm : [#users=1] = call_function[target=torch.mm](args = (%l_x_, %l_y_), kwargs = {})