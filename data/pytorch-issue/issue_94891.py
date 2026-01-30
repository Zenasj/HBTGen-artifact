from transformers import pipeline
generator = pipeline(model="distilgpt2", device=0)
print(generator("Hello github"))

import torch
import ctypes
for n in range(10):
  x = torch.full((1, n+1), 2.).cuda()
  s = torch.sum(x).item()
  y = bin(ctypes.c_uint32.from_buffer(ctypes.c_float(s)).value)  # show value at address of s as int
  print(x)
  print('sum {}'.format(torch.sum(x)))
  print('binary {}\n'.format(y))
  del x, s, y

import torch
import ctypes
for n in range(10):
  x = torch.full((1, n+1), 2.).cuda()
  s_ = torch.sum(x)  # only call once
  s = s_.item()
  y = bin(ctypes.c_uint32.from_buffer(ctypes.c_float(s)).value)
  print(x)
  print('sum {}'.format(s_))
  print('binary {}\n'.format(y))
  del x, s, s_, y