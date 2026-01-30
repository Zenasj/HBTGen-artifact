import torchvision
import random

import torch
import numpy as np
from torch.cuda.amp.autocast_mode import autocast
from torchvision.models.mobilenetv2 import MobileNetV2, mobilenet_v2

GPU_ID = 0

model = MobileNetV2().cuda(GPU_ID)
# model = mobilenet_v2(pretrained=True).cuda(GPU_ID)

## Does not matter is model pretrained or not. Half precision returns nan.


img = np.random.rand(1, 3, 224, 224)

### Full FP32 precision
img32 = torch.from_numpy(img).float().cuda(GPU_ID)
test32 = model(img32)
featured_test32 = model.features[:7](img32)
print(
    f"FP32 full model: {torch.sum(test32).item()} \t model features: {torch.sum(featured_test32).item()}"
)


### Auto Mixed precision
with autocast():
    testmp = model(img32)
    featured_testmp = model.features[:7](img32)
    print(
        f"AMP full model: {torch.sum(testmp).item()} \t model features: {torch.sum(featured_testmp).item()}"
    )


### Half FP16 precision
img16 = torch.from_numpy(img).half().cuda(GPU_ID)
model.half()
test16 = model(img16)
featured_test16 = model.features[:7](img16)
print(
    f"FP16 full model: {torch.sum(test16).item()} \t model features: {torch.sum(featured_test16).item()}"
)