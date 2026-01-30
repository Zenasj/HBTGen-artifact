import torch
import numpy as np

a = torch.tensor(-1e+4+0j)

out_np = np.arcsin(a.numpy()) # (-1.5707964+9.903487j)
out_cpu = torch.asin(a) # tensor(0.+infj)
out_gpu = torch.asin(a.cuda()) # tensor(-1.5708+9.9035j, device='cuda:0')

np.testing.assert_allclose(out_np, out_cpu.numpy(), atol=1)

# AssertionError: 
# Not equal to tolerance rtol=1e-07, atol=1

# Mismatched elements: 1 / 1 (100%)
# Max absolute difference: inf
# Max relative difference: nan
#  x: array(-1.570796+9.903487j, dtype=complex64)
#  y: array(0.+infj, dtype=complex64)