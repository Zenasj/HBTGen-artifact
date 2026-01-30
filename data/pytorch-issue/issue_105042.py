import torchvision

import torch
from torchvision.models import resnet18

torch.set_default_device('cuda')

model = resnet18()
optim = torch.optim.Adam(model.parameters(), lr=0.01)
inp = torch.randn(1, 3, 224, 224)

@torch.compile
def train_step():
    model(inp).sum().backward()
    optim.step()

train_step()
train_step()