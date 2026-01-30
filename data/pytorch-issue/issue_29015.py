import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

from apex import amp


net = torchvision.models.resnet18()
net.cuda()
optim = torch.optim.SGD(net.parameters(), lr=1e-3,)
net, optim = amp.initialize(net, optim, opt_level="O0")

net = copy.deepcopy(net)
net.train()

bn_modules = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)
bn_layers = [
    m
    for m in net.modules()
    if m.training and isinstance(m, bn_modules)
]

print(bn_layers[0].running_mean[:3])
inten = torch.randn(32, 3, 224, 224).cuda()
out = net(inten)
print(bn_layers[0].running_mean[:3])
inten = torch.randn(32, 3, 224, 224).cuda()
out = net(inten)
print(bn_layers[0].running_mean[:3])