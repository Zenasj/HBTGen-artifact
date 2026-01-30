import torch
import torch.nn as nn

torch.nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=5, device=torch.device('cuda'))(torch.randn((1,3,3200,3200), device=torch.device('cuda')))