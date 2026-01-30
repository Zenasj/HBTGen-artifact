import math
import torch
import torch.nn as nn
import os
import numpy as np
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch._dynamo
torch._dynamo.config.suppress_errors = False

class my_modules(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_embedder = PatchEmbed(90, 2, 8, 1152, bias=True, strict_img_size=False)
    
    def forward(self, x):
        x = self.x_embedder(x)
        return x
    
model = my_modules().cuda()
input = torch.rand(80, 8, 102, 76).cuda()

com_model = torch.compile(model)
output = com_model(input)