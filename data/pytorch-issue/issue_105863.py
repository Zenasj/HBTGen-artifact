import torch.nn as nn

import torch
from torch import nn
from torch.optim import Adam


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                                     nn.ReLU(),
                                     nn.Conv2d(16, 32, 3, 2, 1),
                                     nn.AdaptiveAvgPool2d(1))

        self.return_input = True

    def forward(self, x):
        ret = self.layers(x)[0, 0]

        if self.return_input:
            return ret, x
        else:
            return ret


if __name__ == '__main__':
    loss = nn.CrossEntropyLoss()
    net = SimpleNetwork().to('cuda')
    net_optim = torch.compile(net)
    optim = Adam(net_optim.parameters(), lr=1e-4)

    for i in range(10):
        optim.zero_grad()
        dummy_input = torch.rand((1, 3, 32, 32), device=torch.device('cuda', 0))
        dummy_target = torch.randint(0, 32, (1, ), device=torch.device('cuda', 0))
        # here the output is a tuple of (result, input)
        out = net_optim(dummy_input)
        l = loss(out[0], dummy_target)
        l.backward()
        optim.step()

    net_optim.return_input = False
    dummy_input = torch.rand((1, 3, 32, 32), device=torch.device('cuda', 0))
    # the expected output here is just the result. This should NOT be a tuple because we set return_orig to False
    out = net_optim(dummy_input)
    print(type(out))