import torchvision
import torch

from torch.cuda.amp import GradScaler, autocast

model = torchvision.models.vit_b_16().to("cuda")
model.eval()
with torch.no_grad():
    with autocast():
        y_vert_pred = model(torch.randn(16, 3, 224, 224).to("cuda"))

model = torchvision.models.resnet50().to("cuda")
model.eval()
with torch.no_grad():
    with autocast():
        y_vert_pred = model(torch.randn(16, 3, 224, 224).to("cuda"))

model = torchvision.models.vit_b_16().to("cuda")
model.eval()

with autocast():
    y_vert_pred = model(torch.randn(16, 3, 224, 224).to("cuda"))

import torchvision
import torch

from torch.cuda.amp import GradScaler, autocast

model = torchvision.models.vit_b_16().to("cuda")

with autocast():
  with torch.no_grad():
    y_vert_pred = model(torch.randn(16, 3, 224, 224).to("cuda"))