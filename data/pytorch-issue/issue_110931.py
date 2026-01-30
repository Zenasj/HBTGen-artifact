import torch.nn as nn

py
import os
import tempfile

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CyclicLR


model = nn.Linear(100, 100)
opt = SGD(model.parameters(), lr=1.)
scheduler = CyclicLR(opt, base_lr=0.1, max_lr=0.2, scale_fn=lambda x: 0.99)

tmp = tempfile.NamedTemporaryFile(delete=False)
try:
    torch.save(scheduler.state_dict(), tmp.name)
    scheduler.load_state_dict(torch.load(tmp.name))
finally:
    tmp.close()
    os.unlink(tmp.name)