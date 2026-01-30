import torch.nn as nn

import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h

if __name__ == '__main__':
    my_cell = MyCell()
    x = torch.rand(3, 4)
    h = torch.rand(3, 4)
    model = torch.jit.trace(my_cell, (x, h))
    model.save("traced_nn.pt")