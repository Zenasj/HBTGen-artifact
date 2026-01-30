import os

import torchvision.models as models
import torch
from tqdm import tqdm


print(os.getpid())

model = models.resnet18().to('mps')
inputs = torch.randn(5, 3, 224, 224).to('mps')

with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
    for i in tqdm(range(1000)):
        output = model(inputs)