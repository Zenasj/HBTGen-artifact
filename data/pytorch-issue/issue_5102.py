import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(7)

b = 4
ch = 3

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch)
        self.affine = nn.Linear(ch + 1, 2)
        
    def forward(self, x, f):
        x = self.bn(x)
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.view(x.size(0), -1)
        print('x:\n',x)
        print('f:\n',f)
        x = torch.cat([x, f], -1)
        print('x(cat):\n',x)
        out = self.affine(x)
        return out

model = Model()
criterion = nn.CrossEntropyLoss()

model.cuda().train()

x = torch.FloatTensor(b, ch, 32, 32).uniform_()
f = torch.FloatTensor(b).uniform_()
x = Variable(x.float()).cuda()
f = Variable(f).cuda()

y = torch.LongTensor(np.random.randint(0, 2, b))
y = Variable(y).cuda()

out = model(x, f)
print('out:\n%s' % out)
loss = criterion(out, y)
loss.backward()  # crash