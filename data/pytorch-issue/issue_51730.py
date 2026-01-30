import torch.nn as nn

py
import torch
# Define test model
class foo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = torch.nn.Linear(2, 2)
        self.linear_layer2 = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)
    def forward(self, x, y):
        x = self.linear_layer1(x)
        y = self.linear_layer2(y)
        for i in range(2):
            print(x.dtype, y.dtype)
            x = torch.zeros_like(x).scatter_add(0, torch.zeros_like(x, dtype=torch.int64), y)
            x = self.norm(x)
        return x
# Run forward
model = foo().cuda()
x = torch.ones(2,2).cuda()
y = torch.ones(2,2).cuda()
with torch.cuda.amp.autocast(enabled=True):
    res = model(x, y)