import torch.nn as nn

import torch


class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x):
        name_params = self.linear.named_parameters()
        for name, param in name_params:
            x += param

        return x


m = torch.jit.script(TestModule())