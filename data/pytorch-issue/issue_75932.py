import gc
import torch

gc.disable()
cpp_ref = torch.zeros(2, 2)
val = torch.zeros(2, 2)
a = [None]
b = [None, None]
a[0] = b
b[0] = a
b[1] = val
cpp_ref.grad = val
del a, b, val

import gc; gc.collect()
print(cpp_ref.grad)

A = [None]
B = [None, None]
c = torch.zeros([2])
A[0] = B
B[0] = A
B[1] = c