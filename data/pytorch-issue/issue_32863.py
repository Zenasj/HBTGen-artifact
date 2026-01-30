import torch

a = torch.load('bug.pt', 'cuda') # giant uint8 tensor with binary 0/1 values
C = 11359
b = a[:, 0, C]

print('shape', b.shape)
print('argmax', b.argmax())
print('max', b.max())

K = int(b.argmax())

print(b[K]) # should be 1, but is 0
print(a[K, 0, C]) # should be 1, but is 0

print(b.nonzero()) # different result compared to argmax
print('bad argmax', a.argmax(dim = 0)[0][C])