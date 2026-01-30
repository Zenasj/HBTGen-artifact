import torch
import torch.nn as nn
gg = nn.BatchNorm2d(64).eval()
prob = torch.empty(size=(0, 64, 112, 112))
print(gg(prob))