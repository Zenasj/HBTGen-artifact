import torch
import torch.nn as nn
torch.manual_seed(0)

conv = nn.Conv1d(1, 65537, 3, padding=1)

x = torch.ones([1, 1, 3])
y_cpu = conv.to("cpu")(x.to("cpu"))
y_mps = conv.to("mps")(x.to("mps"))

print(y_cpu)
print(y_mps)
print("Equal:", torch.equal(y_cpu, y_mps.to("cpu")))

import torch
import torch.nn.functional as F
torch.manual_seed(0)

out_channels = 65537

weight = torch.randn(out_channels, 1, 1)
x = torch.ones([1, 1, 1])
print(F.conv1d(x.to('cpu'), weight.to('cpu'))) # tensor([[[-1.126], [-1.152], [-0.251], ..., [ 0.275], [ 0.159], [-0.037]]])
print(F.conv1d(x.to('mps'), weight.to('mps'))) # tensor([[[-0.037], [-1.152], [-0.251], ..., [ 0.275], [ 0.159], [-0.564]]], device='mps:0')