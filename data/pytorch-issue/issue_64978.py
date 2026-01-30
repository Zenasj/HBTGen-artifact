Weight: (out_features,in_features)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        b = torch.ones([10])
        return F.linear(a, b)

net = Net()
a = torch.ones([1,1,10])
out = net(a)
print(out)
torch.onnx.export(net, (a,), "tmp.onnx")