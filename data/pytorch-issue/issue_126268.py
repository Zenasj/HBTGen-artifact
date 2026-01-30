import torchvision.models as models
import torch
from torch.utils.flop_counter import FlopCounterMode

device = 'cuda'

inp = torch.randn(1, 3, 224, 224, device = device)
mod = models.resnet18().to(device)

@torch.inference_mode(True)
def run_batch(x):
    flop_counter = FlopCounterMode(mod)
    with flop_counter:
        return mod(x).sum()

run_batch(inp)

import torchvision.models as models
import torch
from torch.utils.flop_counter import FlopCounterMode

device = 'cuda'

inp = torch.randn(1, 3, 224, 224, device = device)
mod = models.resnet18().to(device)

@torch.no_grad()
def run_batch(x):
    flop_counter = FlopCounterMode(mod)
    with flop_counter:
        return mod(x).sum()

run_batch(inp)

import torchvision.models as models
import torch
from torch.utils.flop_counter import FlopCounterMode

device = 'cuda'

inp = torch.randn(1, 3, 224, 224, device = device)
mod = models.resnet18().to(device)

def run_batch(x):
    flop_counter = FlopCounterMode(mod)
    with flop_counter:
        return mod(x).sum()

run_batch(inp)