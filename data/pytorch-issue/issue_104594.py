import torch
import torch.nn as nn

x_shape = [32, 196, 512]
class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(x_shape[-1], x_shape[-1])

    def forward(self, x):
        scale = x.size()[-3] ** -0.5
        x = x * scale
        print(x.dtype)
        return self.linear(x)