import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn
from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Buggy(nn.Module):
    def __init__(self):
        super(Buggy, self).__init__()

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        out = x
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out = F.conv2d(out, weight=torch.randn(out.shape[0], out.shape[1], 1, 1).float().to(device))
        out = out.to(dtype)
        return out


if __name__ == '__main__':
    net = Buggy().to(device)
    inpu = torch.ones([1, 3, 224, 224]).to(device)
    torch.onnx.export(net, inpu, "demo.onnx", verbose=True)

out = F.conv2d(out, weight=torch.randn(out.shape[0], out.shape[1], 1, 1).float().to(device))

class TransposeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)