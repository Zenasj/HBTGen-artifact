import torchvision

import torch
from torchvision.models import convnext_base


model = convnext_base()
model = model.cuda().half()

model = torch.compile(model, mode="max-autotune")

x = torch.randn((1, 3, 224, 224), device="cuda").half()

with torch.no_grad():
    model(x)