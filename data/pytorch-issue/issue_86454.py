from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
import torch.nn as nn

# Dummy net
net = nn.Linear(10, 10)
opt = Adam(net.parameters())
scheduler = LinearLR(opt, start_factor=0, total_iters=4)
for epoch in range(10):
    scheduler.step()