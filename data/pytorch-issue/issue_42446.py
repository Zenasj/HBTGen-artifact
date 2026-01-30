import torch.nn as nn

import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = torch.nn.ModuleList()

        # first layer is needed to reproduce error
        self.layers += [
            torch.nn.Conv1d(1, 16, 1),
        ]
        self.layers += [
            torch.nn.Conv1d(
                16, 16,
                # kernel_size = 19 is OK, but 21 causes error
                kernel_size=21,
                # without group, no error
                groups=2,
            ),
        ]

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        return x


cnn = CNN()
y = torch.randn(1, 1, 128)
y = cnn(y)
loss = F.mse_loss(y, torch.ones_like(y))
loss.backward()