import torch
import torch.nn as nn
import torchvision

x = test_features.to(torch.float).to(device) 
print(x.dtype, x.shape) # torch.float32 torch.Size([64, 3, 200, 200]
with torch.device(device):
    mobilenet = torchvision.models.mobilenet_v2(weights="DEFAULT")
    mobilenet = mobilenet.features[:-1].eval()
    x = mobilenet(x)
    print(x.dtype, x.shape)
    x = nn.AdaptiveAvgPool1d(1)(x)
    print(x.shape)