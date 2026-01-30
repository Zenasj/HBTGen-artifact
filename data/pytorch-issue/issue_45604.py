import torch.nn as nn

import torch
import torch.nn.functional as F
from torch import nn
import time
import os
import argparse

################# FrozenBatchNorm2d jit version ##############
class FrozenBatchNorm2d_jit(torch.jit.ScriptModule):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d_jit, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    @torch.jit.script_method
    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
####################################################

###### test network with jit ########
class TestNet_jit(torch.jit.ScriptModule):
    def __init__(self):
        super(TestNet_jit,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = FrozenBatchNorm2d_jit(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = FrozenBatchNorm2d_jit(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn3 = FrozenBatchNorm2d_jit(64)
    
    @torch.jit.script_method
    def forward(self,x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.conv3(z)
        z = self.bn3(z)
        return z
###########################

################# FrozenBatchNorm2d ##############
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
####################################################

###### test network ########
class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn2 = FrozenBatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        self.bn3 = FrozenBatchNorm2d(64)
    
    def forward(self,x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.conv2(z)
        z = self.bn2(z)
        z = self.conv3(z)
        z = self.bn3(z)
        return z
###########################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--jit", action="store_true", help="enable JIT")
    args = parser.parse_args()

    N = 6
    C = 3
    H = 800
    W = 1344

    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    
    x_nhwc = torch.rand(N,C,H,W).half().to(device='cuda',memory_format=torch.channels_last)
    if not args.jit:
        net_nhwc = TestNet().half().to(device='cuda', memory_format=torch.channels_last)
        out_nhwc = net_nhwc(x_nhwc)
    else:
        net_nhwc_jit = TestNet_jit().half().to(device='cuda', memory_format=torch.channels_last)
        out_nhwc = net_nhwc_jit(x_nhwc)
    
if __name__ == '__main__':
    main()