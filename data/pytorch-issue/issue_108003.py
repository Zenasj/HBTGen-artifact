import torchvision

import torch
from torchvision.models import resnet50, ResNet50_Weights

MEMORY_FORMAT = None # torch.channels_last
DTYPE = torch.float
DEVICE = "cpu"

image_batch = torch.randn(16, 3, 224, 224,dtype=DTYPE).to(DEVICE, memory_format=MEMORY_FORMAT)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(DEVICE, dtype=DTYPE, memory_format=MEMORY_FORMAT).eval()
opt_model = torch.compile(model, fullgraph=True, mode="reduce-overhead")
with torch.no_grad():
    preds1 = opt_model(image_batch)

import torch

def fn(x, y):
    return x+y

opt_fn = torch.compile(fn, mode="reduce-overhead") 
opt_fn(torch.rand(7), torch.rand(7))