import torch.nn as nn

#!/usr/bin/env python

import os
import torch

# Force determinism
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.putenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.set_deterministic(True)
torch.use_deterministic_algorithms(True)
# Force full-precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Device 0: {}'.format(torch.cuda.get_device_name(0)))
print('Device 1: {}'.format(torch.cuda.get_device_name(1)))

rng0 = torch.Generator(device='cuda:0').manual_seed(43)
rng1 = torch.Generator(device='cuda:1').manual_seed(43)

shape = (1, 3, 128, 128)
noise0 = torch.empty(*shape, device='cuda:0').normal_(generator=rng0)
noise1 = torch.empty(*shape, device='cuda:1').normal_(generator=rng1)

if not torch.allclose(noise0.cpu(), noise1.cpu(), atol=0., rtol=0.):
    print(noise0[...,0,0])
    print(noise1[...,0,0])