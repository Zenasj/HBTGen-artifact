import torch

print("size = 9")
x = torch.arange(1, 10)
print(x)
print(torch.sort(x)) # works as expected

print("-"*10)

print("size = 2^15 - 1")
x = torch.arange(1, 32768)
print(x)
print(torch.sort(x)) # works as expected

print("-"*10)

print("size = 2^15")
x = torch.arange(1, 32769)
print(x)
print(torch.sort(x)) # give incorrect values

print("-"*10)

print("size = 100000")
x = torch.arange(1, 100000)
print(x)
print(torch.sort(x)) # give incorrect values

import torch
x = torch.arange(1, 100000)
print(x)
print(torch.sort(x))