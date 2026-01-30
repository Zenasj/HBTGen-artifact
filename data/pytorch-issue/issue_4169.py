import torch.nn as nn

import torch
from torch.autograd import Variable
embed = torch.nn.EmbeddingBag(5,3)
print(next(embed.parameters()))
embed(Variable(torch.LongTensor([0,1,2,3,4])), Variable(torch.LongTensor([0, 1,1, 3, 3, 4])))

import torch
from torch.autograd import Variable
embed = torch.nn.EmbeddingBag(5,3)
print(next(embed.parameters()))
embed(Variable(torch.LongTensor([0,1,2,3,4])), Variable(torch.LongTensor([0, 1, 5])))