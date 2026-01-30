import torch.nn as nn

import sys
import torch
from torch import nn
import numpy as np
class Bottleneck(nn.Module):   
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x+1

class TempOpModel(nn.Module):
    def __init__(self):
        super(TempOpModel,self).__init__()
        self.bottlenecks = nn.ModuleList([Bottleneck() for _ in range(3)])

    def forward(self,x):
        y = list([x,x])
        y.extend(m(y[-1]) for m in self.bottlenecks) # x为[0,0] y=[[0,0],[0,0],[1,1],[2,2],[3,3]
        # out = y[-1]
        # for m in self.bottlenecks:
        #     out = m(out)
        #     y.append(out)
        # y.extend([m(y[-1]) for m in self.bottlenecks]) # x为[0,0] y=[[0,0],[0,0],[1,1],[1,1],[1,1]
        # input = y[-1]
        # for m in self.bottlenecks:
        #     out = m(input)
        #     y.append(out)
        return y
def test_pytorch():
    torch._dynamo.reset()
    net = TempOpModel()
    net.eval()
    with torch.no_grad():
        net_compile = torch.compile(net)       
        indata0 = torch.zeros((2))       
        result0 = net(indata0)
        result1 = net_compile(indata0)
        print(" net \n",result0,"\n net compile:\n",result1)

if __name__=='__main__':
    test_pytorch()

def forward(self,x):
        y = list([x,x])
        y.extend(m(y[-1]) for m in self.bottlenecks) # x为[0,0] y=[[0,0],[0,0],[1,1],[2,2],[3,3]