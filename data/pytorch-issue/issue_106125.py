import torch.nn as nn

#Minimal repro:
#import torch
#class Object:
#  pass
#
#def accesses(c):
#  print(type(c))
#  print(c.__dict__)
#torch.compile(accesses)(Object())

#More lifelike repro:
import torch
from torch import nn

class Config():
  def __eq__(self, other):
    return self.__dict__ == other.__dict__

class Custom(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.c = c

  def forward(self, c2):
    return 1 if self.c==c2 else 0

c1 = Config()
c2 = Config()
m = Custom(c1)
m(c2) # works
torch.compile(m)(c2) #crashes