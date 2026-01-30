import torch
import torch.nn as nn

class MyCell_1(torch.nn.Module):
    def __init__(self):
        super(MyCell_1, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

mycell_1 = MyCell_1()
scripted = torch.jit.script(mycell_1)