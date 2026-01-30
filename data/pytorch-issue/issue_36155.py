import torch.nn as nn
import random

import torch
import torch.nn.functional as F
import numpy as np
from caffe2.python import core, workspace, model_helper

from timeit import Timer
num = 50
N = 1
C = 32
D = 4
H = 112
W = 112
M = 64
kernel_d = 3
kernel_h = 7
kernel_w = 7
stride_d = 1
stride_h = 2
stride_w = 2
padding_d = 1
padding_h = 3
padding_w = 3

X_np = np.random.randn(N, C, D, H, W).astype(np.float32)
W_np = np.random.randn(M, 4, kernel_d, kernel_h, kernel_w).astype(np.float32)
X = torch.from_numpy(X_np)

conv3d_pt = torch.nn.Conv3d(
    C, M, (kernel_d, kernel_h, kernel_w), stride=(stride_d, stride_h, stride_w),
    padding=(padding_d, padding_h, padding_w), groups=8, bias=False)

conv3d_c2 = core.CreateOperator(
    "Conv3D",
    ["X", "W"],
    ["Y"],
    group=8,
    kernels=[kernel_d, kernel_h, kernel_w],
    strides=[stride_d, stride_h, stride_w],
    pads=[padding_d, padding_h, padding_w, padding_d, padding_h, padding_w],
)

m = model_helper.ModelHelper(name="my_conv3d")
weight = m.param_init_net.ConstantFill([], 'conv_w', shape=[64,4,3,7,7], value=1.3)
m.net.Conv3D(["X", "conv_w"], "conv", group=8, kernels=[3, 7, 7], strides=[1,2,2], pads=[1, 3,3, 1, 3, 3])
workspace.RunNetOnce(m.param_init_net)

workspace.FeedBlob("X", X_np)
workspace.FeedBlob("W", W_np)
workspace.CreateNet(m.net)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv3d = conv3d_pt

    def forward(self, x):
        return self.conv3d(x)

model = ConvNet()

def pt_forward():
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        model(X)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

def c2_model_forward():
    workspace.RunNet("my_conv3d")
    # workspace.RunNetOnce(m.net)
t = Timer("pt_forward()", "from __main__ import pt_forward, X")
print("pt time = {}".format(t.timeit(num) / num * 1000.0))
t = Timer("c2_model_forward()", "from __main__ import c2_model_forward, conv3d_c2")
print("c2 model time = {}".format(t.timeit(num) / num * 1000.0))