import torch.nn as nn
import random

import numpy as np
import torch

m = torch.nn.Linear(2, 2, bias=False)
state_dict = {
    'weight': np.random.randn(2, 2)
}
m.load_state_dict(state_dict)