import torch.nn as nn

# loop.py

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
    def forward(self, xs):
        h = torch.zeros(3, 4)
        for i in range(xs.size(0)):
            h = torch.tanh(h + self.linear(xs[i]))
        return h

mymod = MyModule()
x = torch.rand(10, 3, 4)

y = mymod(x)
y.sum().backward()
print(mymod.linear.weight.grad)
mymod.linear.weight.grad.zero_()

traced_mod = torch.jit.trace(mymod, (x,))
y = traced_mod(x)
y.sum().backward()
print(traced_mod.linear.weight.grad)
traced_mod.linear.weight.grad.zero_()

script_mod = torch.jit.script(mymod, (x,))
y = script_mod(x)
y.sum().backward()
print(script_mod.linear.weight.grad)
print(script_mod.code)
script_mod.linear.weight.grad.zero_()

# loop.py

import torch

class MyModuleOriginal(torch.nn.Module):
    def __init__(self, batch_size=10):
        super(MyModuleOriginal, self).__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.batch_size = batch_size
    def forward(self, xs):

        h = torch.zeros(3, 4)
        for i in range(xs.size(0)):
            h = torch.tanh(h + self.linear(xs[i]))
        return h


class MyModuleFixed(torch.nn.Module):
    def __init__(self, batch_size=10):
        super(MyModuleFixed, self).__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.batch_size = batch_size
    def forward(self, xs):

        h = torch.zeros(3, 4)
        for i in range(10):
            h = torch.tanh(h + self.linear(xs[i]))
        return h


x = torch.rand(10, 3, 4)
mymod = MyModuleOriginal()

script_mod = torch.jit.script(mymod)
y = script_mod(x)
y.sum().backward()
print(script_mod.linear.weight.grad)
script_mod.linear.weight.grad.zero_()

y = mymod(x)
y.sum().backward()
print(mymod.linear.weight.grad)
mymod.linear.weight.grad.zero_()

traced_mod = torch.jit.trace(mymod, (x))
y = traced_mod(x)
y.sum().backward()
print(traced_mod.linear.weight.grad)
traced_mod.linear.weight.grad.zero_()

print(script_mod.code)
print("-------")

mymod = MyModuleFixed(x.shape[0])
script_mod = torch.jit.script(mymod)
y = script_mod(x)
y.sum().backward()
print(script_mod.code)
print(script_mod.linear.weight.grad)
script_mod.linear.weight.grad.zero_()

y = mymod(x)
y.sum().backward()
print(mymod.linear.weight.grad)
mymod.linear.weight.grad.zero_()

traced_mod = torch.jit.trace(mymod, (x))
y = traced_mod(x)
y.sum().backward()
print(traced_mod.linear.weight.grad)