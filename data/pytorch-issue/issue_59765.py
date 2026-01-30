import torch
import torch.nn as nn
import math

e = torch.Tensor([[math.log(3/4), math.log(1/4)]])
t = torch.LongTensor([0])
torch.nn.NLLLoss()(e, t)

t = torch.CharTensor([0])
torch.nn.NLLLoss()(e, t) # RuntimeError: expected scalar type Long but found Char

t = torch.ByteTensor([0])
torch.nn.NLLLoss()(e, t) # RuntimeError: expected scalar type Long but found Byte