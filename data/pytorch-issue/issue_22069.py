import torch
import torch.nn as nn
from torch import onnx
from torch.autograd import Variable 

class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x, noise=None):
        shape = x.size()
        noise = torch.randn(shape)
        return noise

model = NoiseLayer()
output = model.forward(Variable(torch.randn(3,256,256)))
print(output)
# Translate this model to ONNX
onnx.export(model, Variable(torch.randn(3,256,256)), '/tmp/model.onnx')