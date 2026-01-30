import torch
import torch.nn as nn

net = nn.Linear(5,1).cuda()
optimizer = torch.optim.Adam(net.parameters())
scaler = torch.cuda.amp.GradScaler(growth_interval=1)
for t in range(1000):
    optimizer.zero_grad()
    x = torch.randn(1,5).cuda()
    y = 0.0001 * torch.randn(1,1).cuda()
    l = ((net(x)-y)**2).mean()
    scaler.scale(l).backward()
    scaler.step(optimizer)
    scaler.update()
    print(scaler._scale)