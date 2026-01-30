import numpy as np
import torch

m1 = np.array([[1,0],[0,1]], dtype=np.float32)
m2 = np.array([[1,2],[3,4]], dtype=np.float32)

print('result np matmul:')
print(np.matmul(m1,m2))

m1_t = torch.from_numpy(m1)
m2_t = torch.from_numpy(m2)

print('result torch matmul cpu:')
print(torch.matmul(m1_t,m2_t))

m1_t = m1_t.cuda()
m2_t = m2_t.cuda()

print('m1_t is cuda:', m1_t.is_cuda)
print('m2_t is cuda:', m2_t.is_cuda)

print('result torch matmul gpu:')
print(torch.matmul(m1_t,m2_t))