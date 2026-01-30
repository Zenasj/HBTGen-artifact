import random

import numpy as np
import torch
np.random.seed(0)

x = -np.ones(3)
y = np.random.randn(3) + 1j * np.random.randn(3)
g = np.random.randn(3) + 1j * np.random.randn(3)

print("x: ", x)
print("y: ", y)
print("============================")

device = torch.device("cpu:0")
x1 = torch.tensor(x, device=device, requires_grad=True)
y1 = torch.tensor(y, device=device, requires_grad=True)
g1 = torch.tensor(g, device=device)
def test():
  tmp_result= torch.Tensor.pow(x1, y1)
  # tmp_result= torch.pow(x1, y1)
  return tmp_result
o1 = test()
o1.backward(g1)
print('nan backward: ',y1.grad.detach().cpu().numpy())