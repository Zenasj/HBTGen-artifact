# both backends have the problem
# import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

import torch

copy_stream = torch.cuda.Stream()
A = torch.ones(1024 ** 3, pin_memory=True)
with torch.cuda.stream(copy_stream):
    B = torch.zeros(1024 ** 3, device='cuda:0')
print(B.data_ptr())
torch.cuda.synchronize()

for _ in range(100): # computation that takes a long time
    B.mul_(2)
del B

# torch.cuda.synchronize() # this synchronization does help, but unable to parallelize multistream 

with torch.cuda.stream(copy_stream): # trying to parallelize the data transfer and computation
    C = A.to('cuda:0', non_blocking=True) # C may use the same memory as B, why?

torch.cuda.synchronize()
print(C.data_ptr()) # B.data_ptr() == C.data_ptr()
print(C) # 6.3383e+29 or something else, should be 1