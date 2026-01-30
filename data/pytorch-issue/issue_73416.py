if (hidden_input.head.abs() > 1000).any:
    m = hidden_input.head.abs().max()
    print('max: {}'.format(m))
    print('The huge number is: {}'.format(hidden_input.head[hidden_input.head.abs() > 1000]))
    print('found huge numbers!')

import torch
from abc import ABC, abstractmethod

class Domain(ABC):

    @abstractmethod
    def __init__(self, head, errors):
        pass

    @abstractmethod
    def matmul(self, other):
        pass

class Zonotope(Domain):

    def __init__(self, head, errors, head2=None, errors2=None):
        self.head = head
        self.errors = errors
        self.head2 = head2
        self.errors2 = errors2

    def matmul(self, other): 
        if isinstance(other, torch.Tensor):
            return self.new(self.head.matmul(other), self.errors.matmul(other),
                            head2=None if self.head2 is None else self.head2.matmul(other),
                            errors2=None if self.errors2 is None else self.errors2.matmul(other))
        else:
            raise Exception("Not supported")

import torch
import torch.nn as nn
import ai

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=1, batch_first=False, nonlinearity=activation)
        self.out = nn.Linear(hidden_size, output_size)
        self.num_neurons = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.name = 'rnn'

    def forward(self, X, dom=None, eps=None, rm=None, min=None, max=None): 
        W_ih = self.rnn.weight_ih_l0.transpose(0, 1)
        W_hh = self.rnn.weight_hh_l0.transpose(0, 1)
        b_ih = self.rnn.bias_ih_l0
        b_hh = self.rnn.bias_hh_l0
        W_o = self.out.weight.transpose(0, 1)
        b_o = self.out.bias

        def forward_layer(input, hidden_input, bs):
            h_i = hidden_input
            x = input

            hx = x.matmul(W_ih)
            hi = h_i.matmul(W_hh)
            h_ = hx + hi + b_ih + b_hh

            h_t = h_.tanh()

            return h_t

        hidden_input = torch.zeros([X.shape[1], hs]).to(device)

        time_step = X.shape[0]
        for i in range(time_step):
            bs = rm[i].item()
            l = torch.clamp(X[i,:]-eps.unsqueeze(1).expand_as(X[i,:]), min=min)
            u = torch.clamp(X[i,:]+eps.unsqueeze(1).expand_as(X[i,:]), max=max)
            head = (l+u)*0.5
            errors = torch.diag_embed((u-l)*0.5)
            dom = ai.Zonotope(head, errors).checkSizes()
            hidden_input = forward_layer(dom, hidden_input, bs)
            # debug code below
            if (hidden_input.head.abs() > 1000).any:
                m = hidden_input.head.abs().max()
                print('max: {}'.format(m))
                print('The huge number is: {}'.format(hidden_input.head[hidden_input.head.abs() > 1000]))
                print('found huge numbers!')

        out = hidden_input.matmul(W_o) + b_o
        return out