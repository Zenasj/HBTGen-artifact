import torch.nn as nn

py
import torch
torch.manual_seed(420)

# model 
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand(5, 4).cuda()

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight)
        z = y.permute(0, 2, 1)
        return z

x = torch.randn(2, 3, 4).cuda()

model = model()
print(model(x))
# works

model_ = torch.compile(model)
print(model_(x))
# KeyError: 'bias'