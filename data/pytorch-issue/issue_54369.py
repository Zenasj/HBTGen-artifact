import torch.nn as nn

import torch
from torch import nn

torch.manual_seed(123)
model = nn.Linear(10,20,bias=True)
x= torch.rand(4,1,10)
print((model(x)[:,0] == model(x[:,0])).all())

model = nn.Linear(512,12800,bias=True)
x= torch.rand(4,1,512)
print((model(x)[:,0] == model(x[:,0])).all())


model = nn.Linear(512,12800,bias=False)
print((model(x)[:,0] == model(x[:,0])).all())

tensor(True)
tensor(False)
tensor(True)