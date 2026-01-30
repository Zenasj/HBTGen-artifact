import torch
from torch.utils.cpp_extension import load


__C = load(
    'sandbox',
    ['sandbox.cpp'],
    verbose=True)


x = torch.ones((10,2), requires_grad=True, device="cpu")*2
y = __C.test_f(x)

print(y)