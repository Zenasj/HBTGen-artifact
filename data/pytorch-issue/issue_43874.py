import math

import torch
import numpy as np

print("python -3//2 =", -3 // 2)
print("numpy -3//2 =", np.array((-3)) // 2)
print("pytorch -3//2 =", torch.tensor([-3]) // 2)

# output:
# python -3//2 = -2
# numpy -3//2 = -2
# pytorch -3//2 = tensor([-1])

y[..., N // 2:] += X

y[..., torch.div(N, 2, rounding_mode='floor'):] += X

import torch_tensorrt
print(torch_tensorrt.__version__)

import torch
import torch.nn as nn
print(torch.__version__)

class TestDiv(nn.Module):
    def forward(self, x):
        x = torch.div(x, 2, rounding_mode="floor")
        return x
    
trt_model = torch_tensorrt.compile(TestDiv(), 
    inputs= [torch_tensorrt.Input((1, 1))]
)

def forward(self, x):
  x = math.floor(torch.div(x, 2))
  return x

# ...

print(trt_model)
# RecursiveScriptModule(original_name=TestDiv_trt)