import torch

a=torch.rand(size=(100000,1,100), dtype=torch.float16, device='cuda')
b=torch.rand(size=(100000,100,100), dtype=torch.float16, device='cuda')
 
torch.cuda.synchronize()
 
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
 
start.record()
z = torch.matmul(a,b)
end.record()
 

torch.cuda.synchronize()
 
print(start.elapsed_time(end))
 
 
a=torch.rand(size=(200000,100), dtype=torch.float16, device='cuda')
b=torch.rand(size=(100,100), dtype=torch.float16, device='cuda')
 
torch.cuda.synchronize()
 
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
 
start.record()
z = torch.matmul(a,b)
end.record()
 

torch.cuda.synchronize()
 
print(start.elapsed_time(end))

In [10]: a = torch.rand(size=(100000,1,100), dtype=torch.float16)

In [11]: b = torch.rand(size=(100000,100,100), dtype=torch.float16)

In [12]: b.shape
Out[12]: torch.Size([100000, 100, 100])

In [13]: torch.matmul(a, b).shape
Out[13]: torch.Size([100000, 1, 100])

In [14]: c = torch.rand(size=(200000,100), dtype=torch.float16)

In [15]: d = torch.rand(size=(100,100), dtype=torch.float16)

In [16]: torch.matmul(c, d).shape
Out[16]: torch.Size([200000, 100])