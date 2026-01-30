import torch.nn as nn

# -*- coding: utf-8 -*-
# @Time    : 2020/11/2
# @Author  : Lart Pang
# @File    : test_amp_ck.py
# @Project : MyProj
# @GitHub  : https://github.com/lartpang
import apex
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


class MyNetWithCheckpoint(torch.nn.Module):
    def __init__(self):
        super(MyNetWithCheckpoint, self).__init__()
        self.dummy_tensor = torch.ones(1, dtype=torch.float32,
                                       requires_grad=True)
        self.in_conv = torch.nn.Conv2d(3, 10, 3, 1, 1)
        self.out_conv = torch.nn.Conv2d(10, 1, 1)

    def forward(self, x):
        x = checkpoint(self.ck_in_conv, x, self.dummy_tensor)
        x = checkpoint(self.ck_out_conv, x)
        return x

    def ck_in_conv(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.in_conv(x)
        return x

    def ck_out_conv(self, x):
        # assert dummy_arg is not None
        x = self.out_conv(x)
        return x


def cal_loss(pred, gt):
    pred = pred.sigmoid()
    loss = (pred - gt) ** 2
    return loss.mean()


mynet = MyNetWithCheckpoint().cuda()

optimizer = torch.optim.SGD(mynet.parameters(), lr=0.001)
USE_AMP = True
ITER_NUM = 10

in_data = torch.randn(4, 3, 20, 20).cuda()
in_gt = torch.randint(low=0, high=2, size=(4, 3, 20, 20)).float().cuda()

# with torch.cuda.amp
scaler = torch.cuda.amp.GradScaler(USE_AMP)

for _ in tqdm(range(ITER_NUM), total=ITER_NUM):
    with torch.cuda.amp.autocast(USE_AMP):
        pred = mynet(in_data)
    loss = cal_loss(pred=pred, gt=in_gt)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

# with apex.amp
mynet, optimizer = apex.amp.initialize(mynet, optimizer, opt_level="O1")

for _ in tqdm(range(ITER_NUM), total=ITER_NUM):
    pred = mynet(in_data)
    loss = cal_loss(pred=pred, gt=in_gt)
    if USE_AMP:
        with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    optimizer.zero_grad()