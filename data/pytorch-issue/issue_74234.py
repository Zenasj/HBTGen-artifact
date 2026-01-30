py
import torch

arg1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)
arg2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float64)
arg3 = torch.tensor([[9, 10]],        dtype=torch.complex128)

res1 = torch.addcmul(arg1, arg2, arg3)
res2 = arg1 + arg2 * arg3

print(res1)
print(res2)
print(res1 == res2)

res1 = torch.addcdiv(arg1, arg2, arg3)
res2 = arg1 + arg2 / arg3

print(res1)
print(res2)
print(res1 == res2)

import torch
arg_1 = torch.rand([5, 5], dtype=torch.float64)
arg_2 = torch.rand([5, 5], dtype=torch.float64)
arg_3 = torch.rand([1, 5], dtype=torch.complex128)
res = torch.addcmul(arg_1,arg_2,arg_3)

'/opt/conda/bin/python', 'test_type_promotion.py', '-v', '--import-slow-tests', '--import-disabled-tests'