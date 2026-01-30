import torch.nn as nn
import random

import numpy as np
import torch

input = torch.tensor(np.random.rand(10, 3, 20, 20))
kernel_size = 4
torch.nn.functional.avg_pool2d(input, kernel_size, stride=0)