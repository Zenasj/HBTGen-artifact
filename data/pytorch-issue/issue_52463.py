import torchvision

import torch, torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

m = torchvision.models.mobilenet_v3_large()
m = torch.jit.script(m)
m = optimize_for_mobile(m)

# run forward 3 times until segfault
m(torch.rand(1, 3, 224, 224))
m(torch.rand(1, 3, 224, 224))
m(torch.rand(1, 3, 224, 224))