import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

class MILSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MILSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter("bias", None)
        self.alpha = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_h = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_i = torch.nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # TODO: type annotations for traced modules
    def forward(self, x, hx, cx):
        # get prev_t, cell_t from states
        Wx = F.linear(x, self.weight_ih)
        Uz = F.linear(hx, self.weight_hh)

        # Section 2.1 in https://arxiv.org/pdf/1606.06630.pdf
        gates = self.alpha * Wx * Uz + self.beta_i * Wx + self.beta_h * Uz + self.bias

        # Same as LSTMCell after this point
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy


class MILSTMEncoder(torch.jit.ScriptModule):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0.0):
        super(MILSTMEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.initial_hidden = torch.nn.Parameter(torch.randn(hidden_size))
        self.initial_cell = torch.nn.Parameter(torch.randn(hidden_size))

        self.milstm = torch.jit.trace(torch.rand(5, hidden_size), self.initial_hidden.repeat(5, 1), self.initial_cell.repeat(5, 1))(MILSTMCell(hidden_size, hidden_size))

    @torch.jit.script_method
    def forward(self, input, seq_lens):
        input = self.embedding(input)
        hidden = self.initial_hidden.unsqueeze(0).repeat([input.size(1), 1])
        cell = self.initial_cell.unsqueeze(0).repeat([input.size(1), 1])
        outputs = torch.zeros([0, input.size(1), input.size(2)])
        for i in range(input.size(0)):
            hidden, cell = self.milstm(input[i], hidden, cell)
            outputs = torch.cat((outputs, hidden.unsqueeze(0)), dim=0)

        return outputs, hidden

milstm = MILSTMEncoder(C, embedding)

inputs = torch.LongTensor(20, 5).random_(0, 10000)
seq_lens = torch.LongTensor(-np.sort(-np.random.randint(low=1, high=20, size=(5,))))
outs = milstm(inputs, seq_lens)
for out in outs:
    print(out)