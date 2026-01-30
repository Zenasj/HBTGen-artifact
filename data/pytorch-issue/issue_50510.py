import torch.nn as nn

import torch
import collections

fields = ['a', 'b', 'c']
MyTuple = collections.namedtuple('MyTuple', fields)

a = torch.rand(2 * 3, device=0)
b = torch.rand(2 * 3, device=0)

a_tensors_for_gpu = [a[2 * i : 2 * i + 2].to(0) for i in range(3)]
out3 = MyTuple(a_tensors_for_gpu[0], a_tensors_for_gpu[1], a_tensors_for_gpu[2])
print(out3)
b_tensors_for_gpu = [b[2 * i : 2 * i + 2].to(0) for i in range(3)]
out4 = MyTuple(b_tensors_for_gpu[0], b_tensors_for_gpu[1], b_tensors_for_gpu[2])
print(out4)
outputs = [out3, out4]

from torch.nn.parallel.scatter_gather import gather
out = gather(outputs, 'cpu')
for i, x in enumerate(out):
    print (i, x)