import torch
import torchvision

from torch import nn
from torchvision.models import resnet50

torch.set_default_device('mps')

img = torch.rand(1,3,64,64)
r50 = resnet50()

ccompiled_r50 = torch.compile(r50)
ccompiled_r50(img)