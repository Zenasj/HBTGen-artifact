import torch.nn as nn

import torch

class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.GRU(input_size=10,
                                hidden_size=10,
                                batch_first=True)

    def forward(self, inputs):
        x, _ = self.rnn(inputs)
        return x


torch.jit.trace(RNN(), torch.zeros((10,10,10)))
torch.jit.script(RNN())