import torch.nn as nn

import torch

device = torch.device('cuda:0')

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        res = x / x.mean(-1, keepdim=True)
        res = res.abs()
        return res
    
model = torch.jit.script(Model()).eval()
x = torch.rand(512, 400).to(device)
for i in range(4):
    res = model(x)
    print(res.isfinite().all().item())