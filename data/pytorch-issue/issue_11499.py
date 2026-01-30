import torch.nn as nn

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.h = nn.Linear(size, size)

model = Model(size=3)
state = model.state_dict()

model_new = Model(size=5)
model_new.load_state_dict(state)