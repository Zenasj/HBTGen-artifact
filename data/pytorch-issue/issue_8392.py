import torch
import torch.nn as nn

self.modlist = torch.nn.ModuleList([SomeModule()] * self.num_mods)

self.enc0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
self.enc1 = self.resnet.layer1
self.enc2 = self.resnet.layer2
self.enc3 = self.resnet.layer3
self.enc4 = self.resnet.layer4

from copy import deepcopy

self.enc0 = nn.Sequential(deepcopy(self.resnet.conv1), deepcopy(self.resnet.bn1), deepcopy(self.resnet.relu), deepcopy(self.resnet.maxpool))
self.enc1 = deepcopy(self.resnet.layer1)
...