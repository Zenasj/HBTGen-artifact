import torch
import torchvision

from torch.optim.adam import Adam
from torch.optim.swa_utils import SWALR
from torchvision.models import resnet18

model = resnet18()
optimizer = Adam(model.parameters())
swa_scheduler = SWALR(optimizer, swa_lr=1e-4, anneal_epochs=0)