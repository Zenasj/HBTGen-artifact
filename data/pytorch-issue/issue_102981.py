import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import copy

m = weight_norm(nn.Linear(20, 40), name='weight')
m2 = copy.deepcopy(m)