import torch.nn as nn

import torch
from torch import nn
import numpy as np

shape = (3, 64, 128, 128)
m = nn.Linear(np.prod(shape), 10)
m = torch.jit.script(m)