import torch

emb1, emb2, cdist_grad = torch.load('cdist_grad.pt')

emb1.retain_grad()
d = torch.cdist(emb1, emb2)
d.backward(cdist_grad)

print(emb1.grad[0, 17])

import torch


def cloned_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(-1, True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(-1, True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([x1.mul(-2), x1_norm, x1_pad], -1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], -1)
    result = torch.matmul(x1_, x2_.transpose(-2, -1))
    return torch.sqrt(result.clamp_min(0))


emb1, emb2, cdist_grad = torch.load("cdist_grad.pt")
emb1 = emb1.cpu()
emb2 = emb2.cpu()
cdist_grad = cdist_grad.cpu()
emb1.retain_grad()
d = cloned_cdist(emb1, emb2)
# d = torch.cdist(emb1, emb2)
d.backward(cdist_grad)
print(d[0, 17])
print(emb1.grad[0, 17])

import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, 128)

    def forward(self, x):
        x = self.fc1(x)
        return x

model = Net().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.000001)
for i in range(10):
    x = torch.randn(32, 512, requires_grad=True).cuda()
    output = model(x)

    loss = torch.cdist(output,output)
    print(loss.size())
    loss = torch.mean(loss)
    print(loss)
    loss.backward()
    optimizer.step()