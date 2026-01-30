import torch
import torch.nn as nn
from torch.autograd import Variable


g = nn.GRU(100, 20, bias=False).cuda()
g = nn.utils.weight_norm(g, name='weight_hh_l0')
g = nn.utils.weight_norm(g, name='weight_ih_l0')
g.flatten_parameters()
zero_state = Variable(torch.zeros(1, 32, 20).cuda())

x = Variable(torch.randn(20, 32, 100).cuda())
output, state = g(x, zero_state)

import torch
import torch.nn as nn
from torch.autograd import Variable


g = nn.GRU(100, 20, bias=False)
g = nn.utils.weight_norm(g, name='weight_hh_l0')
g = nn.utils.weight_norm(g, name='weight_ih_l0')
g.flatten_parameters()
zero_state = Variable(torch.zeros(1, 32, 20))

x = Variable(torch.randn(20, 32, 100))
output, state = g(x, zero_state)

import torch
import torch.nn as nn
from torch.autograd import Variable


g = nn.GRU(100, 20, bias=True)
g = nn.utils.weight_norm(g, name='weight_hh_l0')
g = nn.utils.weight_norm(g, name='weight_ih_l0')
g.flatten_parameters()
zero_state = Variable(torch.zeros(1, 32, 20))

x = Variable(torch.randn(20, 32, 100))
output, state = g(x, zero_state)