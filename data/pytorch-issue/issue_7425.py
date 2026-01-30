import torch.nn as nn
import torch
from torch.autograd import grad

class testModule(nn.Module):
    def __init__(self):
        super(testModule, self).__init__()
        self.lin = nn.Linear(10, 1)

    def forward(self, x):
        return self.lin(x)

def test_cpu():
    t = torch.ones([1, 10], requires_grad=True)
    mod = testModule()
    output = mod(t)
    output[0].backward()
    test = t.grad
    return test

def test_gpu():
    mod = testModule().cuda()
    t = torch.ones([1, 10], requires_grad=True).cuda()
    output = mod(t)
    output[0].backward()
    test = t.grad
    return test

print(test_cpu())

print(test_gpu())

def test_gpu():
    mod = testModule().cuda()
    t = torch.ones([1, 10], requires_grad=True, device="cuda:0")
    output = mod(t)
    output[0].backward()
    test = t.grad