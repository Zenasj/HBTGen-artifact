import torch.nn as nn
import random

import numpy as np
import torch
from torch import nn
import torch.nn.init as init

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def model():
    return double_conv(3,4,4)

if __name__ == '__main__':
    model1 = model()
    model2 = model()
    model3 = model()
    init_weights(model1.conv)
    model2.load_state_dict(model1.state_dict())
    model3.load_state_dict(model1.state_dict())
    dummy = torch.tensor(np.random.uniform(size=(2,3,64,64)).astype(np.float32))

    model1 = model1.to(0)
    model2 = model2.to(1)
    model3 = model3.to(1)
    model3 = torch.nn.DataParallel(model3, device_ids=[1,2])
    # model3 = torch.nn.DataParallel(model3, device_ids=[0,1])

    out1 = model1(dummy.cuda(0))
    out2 = model2(dummy.cuda(1))
    out3 = model3(dummy.cuda(1))

    diff1 = torch.abs(out1.cpu() - out2.cpu()).max()
    diff2 = torch.abs(out1.cpu() - out3.cpu()).max()
    print(diff1, diff2)