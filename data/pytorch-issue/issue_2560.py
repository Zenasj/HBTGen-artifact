import time
import torch
import numpy
import torch.nn as nn

d = 100
td = 0
for i in range (500):
    a = torch.randn(1,d)
    a_vb = torch.autograd.Variable(a,requires_grad=True)
    for i in range(100):
        fc1 = nn.Linear(d,d)
        a_vb = fc1(a_vb)
    b_vb = a_vb.norm()
    t1 = time.time()
    b_vb.backward()
    t2 = time.time()
    td+= t2-t1
print("time: ", td)