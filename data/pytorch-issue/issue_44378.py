import numpy as np
import scipy.linalg
import torch

# sparse 0-1 matrix with shape [100000, 1024]
# each row contains exactly 4 ones and other zeros
a = np.load("a.npy").astype(np.float32)
# int matrix with shape [100000, 128]
b = np.load("b.npy").astype(np.float32)

x, _, _, _ = scipy.linalg.lstsq(a, b)

# 67480.16036171981
print(((a @ x - b) ** 2).sum(-1).mean())

b = torch.from_numpy(b).cuda()
a = torch.from_numpy(a).cuda()
x, _ = torch.lstsq(b, a)

# 7.4439e+15
print(((a @ x[:a.shape[-1]] - b) ** 2).sum(-1).mean())

# raise RuntimeError: Lapack Error in gels : 
# The 259-th diagonal element of the triangular factor of A is zero 
# at /opt/conda/conda-bld/pytorch_1595629395347/work/aten/src/TH/generic/THTensorLapack.cpp:177
x, _ = torch.lstsq(b.cpu(), a.cpu())