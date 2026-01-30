import torch.nn as nn

import torch
input = torch.rand([1, 1, 2, 2], dtype=torch.float32)
indices = torch.randint(-16,1024,[1, 1, 2, 2], dtype=torch.int64)
kernel_size = [16, -1024]
stride = [-16, 1]
print(torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride))
# tensor([], size=(1, 1, 0, -1023))

import torch
input = torch.rand([1, 1, 2, 2], dtype=torch.float32)
indices = torch.randint(-16,1024,[1, 1, 2, 2], dtype=torch.int64)
kernel_size = [16, -1024]
# stride = [-16, 1]
stride = None
print(torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride))
# RuntimeError: [enforce fail at CPUAllocator.cpp:50] ((ptrdiff_t)nbytes) >= 0. alloc_cpu() seems to have been called with negative number: 18446744073709289472