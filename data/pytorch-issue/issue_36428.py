import torch.nn as nn

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, 2)
        self.lstm = nn.LSTM(input_size=10, hidden_size=10,
                                batch_first=True,bidirectional=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        return x


net = Net().cuda()
net.train()
optimizer = optim.SGD(net.parameters(), lr=1e-3)

scaler = GradScaler()

optimizer.zero_grad()

with autocast():
    output = net(torch.rand(10,1,10).cuda())

import torch

class Net(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, (3, 3), padding=(1, 1))
        self.gru = torch.nn.LSTM(dim, 1024, batch_first=True)

    def forward(self, x):
        out = self.conv(x)
        with torch.cuda.amp.autocast():
            out = out.reshape(-1, out.size(1) * out.size(2), out.size(3))
            out, _ = self.gru(out)
        return out

model = Net(512).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=.9)
scaler = torch.cuda.amp.GradScaler()
optimizer.zero_grad()

x = torch.rand((32, 1, 1600, 512), dtype=torch.float32).cuda()
y = model(x)

import torch

class Net(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, (3, 3), padding=(1, 1))
        self.gru = torch.nn.LSTM(dim, 1024, batch_first=True)

    def forward(self, x):
        with torch.cuda.amp.autocast():
            out = self.conv(x)  # <= here is the change
            out = out.reshape(-1, out.size(1) * out.size(2), out.size(3))
            out, _ = self.gru(out)
        return out

model = Net(512).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=.9)
scaler = torch.cuda.amp.GradScaler()
optimizer.zero_grad()

x = torch.rand((32, 1, 1600, 512), dtype=torch.float32).cuda()
y = model(x)