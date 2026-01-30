import torch
import torch.nn as nn

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.l = nn.Conv2d(1,1,3)
    def forward(self, input):
        return self.l(input)

net = test().cuda()
print(net(torch.rand(1,1,5,5).cuda()))