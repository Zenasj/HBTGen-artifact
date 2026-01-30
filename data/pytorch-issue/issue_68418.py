import torch.nn as nn

nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), 
        nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
)

import torch 
import traceback
from torch import nn

#torch.cuda.empty_cache()
#print(f"Allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")

a = torch.randn(190, 1, 64, 2933)
a = a.cuda()

#print(f"Allocated: {torch.cuda.memory_allocated()}, reserved: {torch.cuda.memory_reserved()}")

net = nn.Sequential(
        nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), 
        nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
        nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(True)
) 
net.cuda()

try:
    net(a)
except: 
    traceback.print_exc()