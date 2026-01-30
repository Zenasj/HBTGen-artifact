import torch
import torch.nn as nn

self.feature =  torch.nn.Linear(7*7*64, 2) # Feature extract layer
self.pred = torch.nn.Linear(2, 10, bias=False) # Classification layer

self.pred.weight = self.pred.weight / torch.norm(self.pred.weight, dim=1, keepdim=True)

self.pred.weight = torch.nn.Parameter(self.pred.weight / torch.norm(self.pred.weight, dim=1, keepdim=True))