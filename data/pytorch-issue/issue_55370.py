# 1.8.1+cpu, python3.7.4, Win10
import torch
from torch.nn import Parameter

T1 = Parameter(torch.rand(3,3,21,2),requires_grad=True)
T2 = Parameter(torch.rand(22,2),requires_grad=False)
dis = torch.cdist(T1,T2)
out = dis.max()
print(out)
out.backward()