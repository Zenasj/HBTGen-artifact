import torch.nn as nn
import random

import torch
import numpy as np

input = torch.tensor((np.random.rand(2,2)*100).astype('int64'))
weight = torch.tensor(1)
torch.nn.functional.embedding(input, weight)