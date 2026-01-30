import torch.nn as nn

class MyLayer(nn.Module):
  def __init__(self):
    self.a = nn.Linear(784,200)
    self.b = nn.Linear(784,20)
  def forward(self,x):
    return self.a(x)