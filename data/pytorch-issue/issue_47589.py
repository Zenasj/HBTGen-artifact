import torch.nn as nn

import torch


weight = torch.nn.Linear(100, 100).to("cuda")


class Foo:
    def __init__(self):
        self.lr = 1.0
        self.op = torch.optim.Adam(weight.parameters())
        self.sc = torch.optim.lr_scheduler.LambdaLR(self.op, lr_lambda=lambda s: self.lr)

    def run(self):
        loss = weight(torch.randn(32, 100).to("cuda")).mean()
        self.op.zero_grad()
        loss.backward()
        self.sc.step()
        self.op.step()


while True:
    Foo().run()

import torch


weight = torch.nn.Linear(100, 100).to("cuda")


class Foo:
    def __init__(self):
        self.lr = 1.0
        self.op = torch.optim.Adam(weight.parameters())
        self.sc = torch.optim.lr_scheduler.LambdaLR(self.op, lr_lambda=lambda s: 1.0)  # no reference to the instance field

    def run(self):
        loss = weight(torch.randn(32, 100).to("cuda")).mean()
        self.op.zero_grad()
        loss.backward()
        self.sc.step()
        self.op.step()


while True:
    Foo().run()

import torch


weight = torch.nn.Linear(100, 100).to("cuda")


while True:
    op = torch.optim.Adam(weight.parameters())
    sc = torch.optim.lr_scheduler.LambdaLR(op, lr_lambda=lambda s: 1.0)
    loss = weight(torch.randn(32, 100).to("cuda")).mean()
    op.zero_grad()
    loss.backward()
    sc.step()
    op.step()