import torch.nn as nn

from torch import nn
import torch._dynamo as dynamo


class fmod(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    self.num_experts = 32
    self.top_k = 4
    routing_weights, selected_experts = torch.topk(x, 4, dim=-1)
    one_hot_encoded = torch.nn.functional.one_hot(selected_experts, 32) #.permute(2, 1, 0)
    return one_hot_encoded.sum()
m = fmod()
x = torch.randn((1,32,32))
explanation= dynamo.explain(f, x)
print(explanation)

import torch
from torch import nn


class fmod(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self,x):
    self.num_experts = 32
    self.top_k = 4
    routing_weights, selected_experts = torch.topk(x, 4, dim=-1)
    one_hot_encoded = torch.nn.functional.one_hot(selected_experts, 32) #.permute(2, 1, 0)
    return one_hot_encoded.sum()
m = fmod()
x = torch.randn((1,32,32))
torch.compile(m, fullgraph=True)(x)