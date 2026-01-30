import torch.nn as nn

import torch
from torch import nn
import torch._dynamo


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        return r_out, h_state

model = RNN().eval()
opt_model = torch._dynamo.optimize("aot_eager")(model)
x = torch.rand([4, 4, 1])
y = torch.rand([1, 4, 32])
print(opt_model(x, y))