import torch

def forward(self, x):
    return x.size(0)

...

out1 = m(torch.randn(2,3))  # 2
out2 = m(torch.randn(3,3))  # 3