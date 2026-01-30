import torch
import torch.nn as nn
from torch.cuda.amp import *
dev = 'cuda'
net = nn.LSTMCell(3, 3, bias=True).to(dev)
x = torch.randn(2, 3).to(dev)
with autocast():    
    h, c = net(x.half())