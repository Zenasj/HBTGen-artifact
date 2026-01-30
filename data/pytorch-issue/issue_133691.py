import torch.nn as nn

import torch

device = torch.device("cuda:0")

# this one works fine
# class Test(torch.nn.Sequential):
#     pass

class Test(torch.nn.Sequential):
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x
    

model = Test(torch.nn.Linear(64, 64)).to(device)
model.compile()

x = torch.randn(4, 16, 64, device=device)
out = model(x)