import torchvision

import torch
from torchvision.models import vgg16

x = torch.zeros((1, 3, 224, 224))
model = vgg16(pretrained=False)
model.cuda()
model(x)