import torchvision
import random

from typing import Tuple
from PIL.Image import Image
from einops import rearrange # pip install einops
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor

tv_ds = CIFAR10(".", download=True)
YCbCr_images: Tuple[Image, ...] = tuple(image.convert("YCbCr") for image, label in tv_ds)
tensors: Tuple[torch.Tensor, ...] = tuple(
    rearrange(to_tensor(image), "c h w -> h w c") for image in YCbCr_images
)
x = torch.stack(tensors).unsqueeze(0)  # 1 × 50000 × 32 × 32 × 3                                                                                                                                                                             
x_flattened = x.reshape(-1, 3)
print(x_flattened.mean(dim=0))  # (0.3277, 0.3277, 0.3277)                                                                                                                                                                                   
for channel in range(3):
    print(x_flattened[:, channel].mean())  # 0.3277, 0.3277, 0.3277 on google colab, but 0.4794, 0.4916, 0.5037 on my local machine                                                                                                                                                                     
print(x_flattened.double().mean(dim=0))  # (0.4790, 0.4809, 0.5077)                                                                                                                                                                          
print(x_flattened.cuda().mean(dim=0))  # (0.4790, 0.4809, 0.5077)

for channel in range(3):
    print(x_flattened[:, channel].clone().mean())

a = numpy.random.rand(51200000, 3)
a = torch.tensor(a, dtype=torch.float32)
b = a.sum(axis=0)
print(b)

a = numpy.random.rand(3, 51200000)
a = torch.tensor(a, dtype=torch.float32)
b = a.sum(axis=1)
print(b)