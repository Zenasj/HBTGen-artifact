import torch.nn as nn

import torch

from torch.optim.lr_scheduler import LambdaLR

params = torch.nn.LSTM(10, 10).parameters()
optimizer = torch.optim.Adam(params)
scheduler = LambdaLR(optimizer, lambda x: 1.0)
state_dict = scheduler.state_dict()
scheduler.load_state_dict(state_dict)
scheduler.load_state_dict(state_dict)