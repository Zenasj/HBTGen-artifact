import torch.nn as nn
import random

#!/usr/bin/env python3

import os, sys
import numpy as np
import torch

cuda_flag = 1

def numpy_to_torch_var(np_t):
    if cuda_flag: t = torch.from_numpy(np_t).cuda()
    else: t = torch.from_numpy(np_t).clone()
    t.requires_grad = True
    return t

prev_layer = numpy_to_torch_var(np.random.randn(1, 84, 84).astype(np.float32))
prev_layer = torch.nn.functional.max_pool2d(prev_layer, kernel_size=(3, 3))

import torch

device = "cuda"

prev_layer = torch.randn(1, 84, 84, device=device)
prev_layer = torch.nn.functional.max_pool2d(prev_layer, kernel_size=(3, 3))
torch.cuda.synchronize()