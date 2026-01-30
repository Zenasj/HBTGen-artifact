import torch
from torch.autograd.gradcheck import gradcheck

torch.set_default_tensor_type("torch.cuda.FloatTensor") # <2.0.0
# torch.set_default_device("cuda") # >2.0.0


x = torch.randn(3, dtype=torch.double, requires_grad=True)

def func(inp):
  return inp ** 2.0

assert gradcheck(func, x, fast_mode=True)