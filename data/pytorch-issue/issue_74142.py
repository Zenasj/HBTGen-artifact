import torch.nn as nn

import torch

N = 1 # can be anything else

class Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x.flatten()

net = Net().eval()

o = net(torch.zeros((N), dtype=torch.float))
print(o.shape)

with torch.no_grad():
  torch.onnx.export(net, (
      torch.ones((N), dtype=torch.float)), "output.onnx", verbose=True, opset_version=14)