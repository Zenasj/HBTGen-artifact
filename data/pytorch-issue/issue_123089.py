import torch
import torch.nn as nn
from torch.export import export

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(800, 800)

    def forward(self, x):
        x = self.rnn(x)
        return x

mod = Model()
mod.eval()
mod.cuda()
exported = export(mod, (torch.rand(20, 800, 800).cuda(),))