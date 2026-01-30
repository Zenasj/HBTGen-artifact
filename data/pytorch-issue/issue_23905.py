import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.x = {}

    def forward(self, input):
        return self.x

scripted_module = torch.jit.script(MyModule())