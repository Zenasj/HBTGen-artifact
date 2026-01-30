import torch.nn as nn

import torch
from torch import nn
from torch.nn import functional as F



class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.lstm = nn.LSTM(10, 20, 2)
        lstm_norm_fn_pntr = nn.utils.weight_norm
        self.lstm = lstm_norm_fn_pntr(self.lstm, "weight_hh_l0")

    def forward(self, x):
        self.lstm.flatten_parameters()
        self.lstm(x)
        return x


if __name__ == "__main__":
    m = MyModule()
    m.cuda()
    m.train()
    for _ in range(10):
        m(torch.randn(1, 10, 10).cuda())