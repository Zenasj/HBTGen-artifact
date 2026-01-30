import torch
import torch.nn as nn

def forward(self, x):
          size_array = [int(s) for s in x.size()[2:]]
          return torch.nn.functional.avg_pool2d(x, size_array)