import torch.nn as nn

import torch
from torch import nn

class Simple2Conv(nn.Module):
    def __init__(self):
        super(Simple2Conv, self).__init__()
        self.layer_1 = nn.Conv3d(1, 1, 3, padding=1)
        self.layer_2 = nn.Conv3d(1, 1, 3, padding=2, dilation=2)

    def forward(self, x):
        layer_1 = self.layer_1(x)        
        layer_2 = self.layer_2(layer_1)  
        print(layer_2.shape)
        return layer_2

model = Simple2Conv().cuda()

isize = 256
osize = isize 
x = torch.FloatTensor(1,1,isize,isize,isize).cuda()

output = model(x)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
y = torch.LongTensor(1,osize ,osize ,osize).cuda()

loss = criterion(output, y)
optimizer.zero_grad()
loss.backward()