import torch.nn as nn

#!/usr/bin/env python

import torch
from torch._dynamo import optimize
torch.manual_seed(0)

class Model(torch.nn.Module):
    def __init__(self, channels):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

n, c, h, w = 8, 640, 16, 16
x = torch.randn((n, c, h, w))

model = Model(c)
jit_model = optimize("inductor")(model)
jit_model(x)

torch.save(jit_model.state_dict(), "model.pt")

# Someone else is trying to load the checkpoint
model = Model(c)
model.load_state_dict(torch.load("model.pt"))