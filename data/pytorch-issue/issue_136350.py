import torch
import time
offset=1726274430
torch.set_printoptions(precision=10)
l1=[time.time()-offset,time.time()-offset]
print('list:', l1, type(l1[0]))
t1=torch.tensor(l1, dtype=torch.double)
print('cpu:', t1, t1.dtype)
t1=t1.to('cuda')
print('gpu:', t1, t1.dtype)