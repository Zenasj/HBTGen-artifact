import torch.nn as nn

class Upsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(2), mode='nearest') 
    
    def forward(self, input, weight):
        weight =  self.upsample(weight)
        return torch.nn.functional.conv2d(input, weight)

import torch
from torch import nn

class Upsampling(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(size=[2,2], mode='nearest')

    def forward(self, input, weight):
        weight =  self.upsample(weight)
        return torch.nn.functional.conv2d(input, weight)

torch.onnx.export(Upsampling(), (torch.tensor([[[[1,2],[1,2]]]], dtype=torch.float32), torch.tensor([[[[1]]]], dtype=torch.float32)), "test.onnx")