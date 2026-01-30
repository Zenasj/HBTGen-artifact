import torch.nn as nn

import torch
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, dilation=1).cuda()
#  fc = torch.nn.Linear(32, 10)
criteria = torch.nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal_(conv1.weight, 1)
torch.nn.init.kaiming_normal_(conv2.weight, 1)
params = list(conv1.parameters()) + list(conv2.parameters())
optim = torch.optim.SGD(params, lr=1e-4, momentum=0.9, weight_decay=1e-4)

for i in range(10):
    inten = torch.randn(16, 3, 64, 64).cuda()
    lb = torch.randint(0, 10, (16, 64, 64)).cuda()

    optim.zero_grad()
    feat = conv1(inten)
    feat = conv2(feat)
    loss = criteria(feat, lb)
    loss.backward()
    optim.step()
    print(loss.item())

import torch
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1).cuda()
criteria = torch.nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal_(conv1.weight, 1)
torch.nn.init.kaiming_normal_(conv2.weight, 1)
params = list(conv1.parameters()) + list(conv2.parameters())
optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)


grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = torch.mean(grad).item()
    return hook

for i in range(10):
    inten = torch.randn(16, 3, 64, 64).cuda()
    lb = torch.randint(0, 10, (16, 64, 64)).cuda()

    optim.zero_grad()
    feat = conv1(inten)
    feat = conv2(feat)
    feat = F.interpolate(feat, inten.size()[2:], mode='bilinear', align_corners=True)
    N, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    feat.register_hook(save_grad('feat'))
    lb = lb.view(-1)
    loss = criteria(feat, lb)
    loss.backward()
    optim.step()

    print(grads['feat'])

import torch
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1).cuda()
criteria = torch.nn.MSELoss()

torch.nn.init.kaiming_normal_(conv1.weight, 1)
torch.nn.init.kaiming_normal_(conv2.weight, 1)
params = list(conv1.parameters()) + list(conv2.parameters())
optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = torch.mean(grad).item()
    return hook


for i in range(10):
    inten = torch.randn(16, 3, 64, 64).cuda()
    lb = torch.rand((16, 32, 64, 64)).cuda()

    optim.zero_grad()
    feat = conv1(inten)
    feat = conv2(feat)
    feat = F.interpolate(feat, inten.size()[2:], mode='bilinear', align_corners=True)
    N, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    feat.register_hook(save_grad('feat'))
    lb = lb.view(-1, C)
    loss = criteria(feat, lb)
    loss.backward()
    optim.step()

    print("{:.20f}".format(grads['feat']))

import torch
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True


conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1).cuda()
criteria = torch.nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal_(conv1.weight, 1)
torch.nn.init.kaiming_normal_(conv2.weight, 1)
params = list(conv1.parameters()) + list(conv2.parameters())
optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = torch.mean(grad).item()
    return hook


for i in range(10):
    inten = torch.randn(16, 3, 64, 64).cuda()
    lb = torch.randint(0, 2, (16, 64, 64)).cuda()

    optim.zero_grad()
    feat = conv1(inten)
    feat = conv2(feat)
    feat = F.interpolate(feat, inten.size()[2:], mode='bilinear', align_corners=True)
    N, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    feat.register_hook(save_grad('feat'))
    lb = lb.view(-1)
    loss = criteria(feat, lb)
    loss.backward()
    optim.step()

    print("{:.20f}".format(grads['feat']))

import torch
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(123)
random.seed(123)
np.random.seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1).cuda()
conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, dilation=1).cuda()
criteria = torch.nn.CrossEntropyLoss()

torch.nn.init.kaiming_normal_(conv1.weight, 1)
torch.nn.init.kaiming_normal_(conv2.weight, 1)
params = list(conv1.parameters()) + list(conv2.parameters())
optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9, weight_decay=1e-4)


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = torch.mean(grad).item()
    return hook


for i in range(10):
    inten = torch.randn(16, 3, 64, 64).cuda()
    lb = torch.randint(0, 2, (16, 32, 32)).cuda()

    optim.zero_grad()
    feat = conv1(inten)
    feat = conv2(feat)
    # feat = F.interpolate(feat, inten.size()[2:], mode='bilinear', align_corners=True)
    N, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
    feat.register_hook(save_grad('feat'))
    lb = lb.view(-1)
    loss = criteria(feat, lb)
    loss.backward()
    optim.step()

    print("{:.20f}".format(grads['feat']))