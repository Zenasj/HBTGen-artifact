import torch
import torch.nn as nn

torch.nn.MaxPool2d(4,stride=4,padding=3)(Variable(torch.rand(1,1,20,20)))