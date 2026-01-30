import torch.nn as nn
import torchvision

import torch
from torchvision.models import resnet18

net = resnet18().cuda()
x = torch.zeros((1, 3, 224, 224)).float().cuda()
y = net(x)

# Could not load library libcudnn_cnn_infer.so.8. Error: libnvrtc.so: cannot open shared object file: No such file or directory
# Aborted (core dumped)

import torch
from torch import nn

net = nn.Sequential(nn.Linear(10,10), nn.Sigmoid(), nn.Linear(10,1)).cuda()
x = torch.zeros((1, 10)).float().cuda()
y = net(x)    # OK