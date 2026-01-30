import torch
import numpy as np
import random

sum((torch.rand(100000000) <= 0.).sum().float() for i in range(20)) / 20

sum((torch.rand(100000000, device=torch.device('cuda')) <= 0.).sum().float() for i in range(20)) / 20

sum((np.random.rand(100000000).astype('float32') <= 0.).sum() for i in range(20)) / 20

sum((torch.rand(100000000, dtype=torch.double).float() <= 0.).sum().float() for i in range(20)) / 20