import torch
torch.ops.load_library("./build/libtest.so")
a = torch.rand([3, 2]).cuda()
b = torch.rand([2, 3]).cuda()
c = torch.rand([8000]).cuda()
print(torch.ops.test.mm(a, b))
print(torch.ops.test.sort(c))

import ctypes
ctypes.CDLL("./build/libtest.so")
import torch
torch.ops.load_library("./build/libtest.so")
a = torch.rand([3, 2]).cuda()
b = torch.rand([2, 3]).cuda()
c = torch.rand([8000]).cuda()
print(torch.ops.test.sort(c))
print(torch.ops.test.mm(a, b))