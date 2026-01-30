import random

import numpy as np
import torch
npz_file = np.load('sum_bug_data.npz')
A, B = npz_file['A'], npz_file['B']

X_np = A
Y_np = np.concatenate([A, B], axis=0)
X = torch.from_numpy(X_np).cuda()
Y = torch.from_numpy(Y_np).cuda()

assert A.shape == (10, 7923, 100)
assert B.shape == (10, 7923, 100)
assert X.shape == (10, 7923, 100)
assert Y.shape == (20, 7923, 100)

X_sum = torch.sum(X, dim=(1,2)) 
Y_sum = torch.sum(Y, dim=(1,2)) 
print('X     ==     Y[:10] -> ', torch.all(X     ==     Y[:10]))
print('X_sum == Y_sum[:10] -> ', torch.all(X_sum == Y_sum[:10]))
print('-'*80)
with np.printoptions(precision=10):
    print('X_sum[:10] ->\n', X_sum[:10].detach().cpu().numpy().reshape(-1))
    print('Y_sum[:10] ->\n', Y_sum[:10].detach().cpu().numpy().reshape(-1))
    print('-'*80)
    print('X_sum[:10] == Y_sum[:10] ->\n', (X_sum[:10] == Y_sum[:10]))

import torch
import numpy as np
import cupy as cp
import time

X_np = np.random.randn(10, 7923, 100).astype(np.float32)
Y_np = np.random.randn(20, 7923, 100).astype(np.float32)

X_tch = torch.from_numpy(X_np).cuda()
Y_tch = torch.from_numpy(Y_np).cuda()
X_cp = cp.asarray(X_np)
Y_cp = cp.asarray(Y_np)

time_cp = -time.time()
for _ in range(1000):
    a = X_cp.sum((1,2))
    b = Y_cp.sum((1,2))
time_cp += time.time()
    
time_tch = -time.time()
for _ in range(1000):
    a = X_tch.sum((1,2))
    b = Y_tch.sum((1,2))
time_tch += time.time()

print('Time to Perform the Same Summations:')
print('  -> PyTorch took %.2f ms.' % (time_tch*1000))
print('  -> CuPy    took %.2f ms.' % (time_cp *1000))