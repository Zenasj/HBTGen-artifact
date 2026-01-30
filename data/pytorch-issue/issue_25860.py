import torch
import torch.nn as nn

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.v = torch.randn(1)

    def update_v(self):
        self.v = torch.randn(1)

    def forward(self):
        self.update_v()  # <--- HERE

model = torch.jit.script(MyModule())