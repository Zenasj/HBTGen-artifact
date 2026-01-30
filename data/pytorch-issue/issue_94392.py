from timm.optim import Lamb

import torch
import torch.nn as nn

class Repro(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def train_step(self, x, optimizer):
        loss = self(x).mean()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    model = Repro().cuda()
    optimizer = Lamb(model.parameters())
    opt = torch.compile(model.train_step, backend='eager')
    data = torch.rand(2, 4).cuda()

    for i in range(2):
        opt(data, optimizer)