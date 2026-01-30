import torch
from torch._subclasses.fake_tensor import FakeTensorMode

torch.cuda.is_available()  # returns False
device = torch.device("cuda:0")

with FakeTensorMode():
   t1 = torch.empty(10, device=device)
   t2 = torch.ones(10, device=device)
   t3 = torch.zeros(10, device=device)
   t4 = torch.rand(10, device=device)
   t5 = torch.tensor([1,2,3], device=device)