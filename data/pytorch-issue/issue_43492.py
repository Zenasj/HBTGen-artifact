import torch

n = int(1e8)
a = torch.randn(n)

# this works
print(a.cuda().expand((n, n)))

# this breaks
print(a.expand((n, n)).cuda())

def to_cuda_with_strides(t):
  cuda_storage = t.storage().cuda()
  result = torch.empty(0, device = cuda_storage.device, dtype = cuda_storage.dtype) 
  result.set_(cuda_storage, t.storage_offset(), size=t.shape, stride=t.stride())                              
  return result