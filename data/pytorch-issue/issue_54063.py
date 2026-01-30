import random

import numpy as np
import scipy.linalg

V = np.random.rand(17, 17)
scipy.linalg.inv(V)

import torch

@torch.jit.script
def jit_test(x):
    1 + 2