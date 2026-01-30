import torch
import torchvision.models as models
import numpy as np
import random

SEED = 0

## FLOAT 32
torch.manual_seed(SEED)
torch.set_default_tensor_type(torch.FloatTensor)

a = torch.randn(1)
print(a) # --> 1.5410
alexnet = models.alexnet(pretrained=False)
for k,v in alexnet.state_dict().items():
    print(k, torch.sum(v)) # --> 3.7782
    break

## FLOAT 64
torch.manual_seed(SEED)
torch.set_default_tensor_type(torch.DoubleTensor)

a = torch.randn(1)
print(a) # --> 1.5410
alexnet = models.alexnet(pretrained=False)
for k,v in alexnet.state_dict().items():
    print(k, torch.sum(v)) # --> 6.9368, but I want this to be ~ 3.7782
    break