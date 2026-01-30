import torch

print("input tensors a, b:")
a = torch.rand([6])
a1 = torch.clone(a)
print(a)
b = torch.rand([6])
b1 = torch.clone(b)
print(b)

def func(x,y):
  x.data = y
  y.data = torch.zeros(0)
  return x

f_compiled = torch.compile(func)
f_compiled(a,b)
print("\n a, b after compile run:")
print(a)
print(b)

func(a1,b1)
print("\n a, b after eager run:")
print(a1)
print(b1)

import torch

print("input tensors a, b:")
a = torch.rand([6])
a1 = torch.clone(a)
print(a)
b = torch.rand([6])
b1 = torch.clone(b)
print(b)

def func(x,y):
  x.data = y  # Commenting out this line gives the right result
  x += 1
  return x

f_compiled = torch.compile(func)
f_compiled(a,b)
print("\n a, b after compile run:")
print(a)
print(b)

func(a1,b1)
print("\n a, b after eager run:")
print(a1)
print(b1)