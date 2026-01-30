py
import torch
q = torch.zeros(5)
w = torch.rand(4)
e = torch.tensor([1,1,1,1])
q[e] += w

py
import torch
q = torch.zeros(5,5)
w = torch.rand(4,5)
e = torch.tensor([1,1,2,2])
q[e,:] += w
q
w

py
import torch
q = torch.zeros(3,5)
w = torch.rand(4,5)
e = torch.tensor([1,1,2,2])
print(q.index_add_(0,e,w))
print(w)