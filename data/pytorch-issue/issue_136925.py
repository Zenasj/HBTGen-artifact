import torch.nn as nn

import torch
from torch import nn

my_tensor = torch.tensor([[8., -3., 0., 1., 5., -2.]])

torch.manual_seed(42)

rnn = nn.RNN(input_size=6, hidden_size=3)
                   # ↓↓↓
rnn(input=my_tensor, h_0=torch.tensor([[0., 1., 2.]])) # Error

import torch
from torch import nn

my_tensor = torch.tensor([[8., -3., 0., 1., 5., -2.]])

torch.manual_seed(42)

rnn = nn.RNN(input_size=6, hidden_size=3)
                   # ↓↓
rnn(input=my_tensor, hx=torch.tensor([[0., 1., 2.]]))
# (tensor([[ 0.9134, -0.8619,  0.9997]], grad_fn=<SqueezeBackward1>),
#  tensor([[ 0.9134, -0.8619,  0.9997]], grad_fn=<SqueezeBackward1>))