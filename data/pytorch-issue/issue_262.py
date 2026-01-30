register_backward_hook('name', hook, write=True)

import torch
from torch.autograd import Variable

x = Variable(torch.randn(5, 5), requires_grad=True)
y = Variable(torch.randn(5, 5), requires_grad=True)

a = x * 2
b = y * 3

def hook_a(grad_output):
    grad_output.mul_(2)

a.register_hook('test', hook_a)

c = a + b
c.sum().backward()

print(x.grad) # should be 2, is 2
print(y.grad) # should be 3, is 6