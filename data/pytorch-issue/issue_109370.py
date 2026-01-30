import torch.nn as nn

import torch
from torch import nn

class demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

    def forward(self, x):
        out = self.avgpool(x)
        return out

if __name__ == '__main__':
    a = torch.randn(1, 3, 12, 12)
    network = demo()
    print(network(a).shape)
    torch.onnx.export(network, a, "demo.onnx", verbose=True)