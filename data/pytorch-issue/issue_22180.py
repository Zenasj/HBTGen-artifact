import torch
import torch.nn as nn

class M(torch.jit.ScriptModule):
    def __init__(self):
        super(M, self).__init__()
        self.conv = nn.Conv2d(5, 5, 2)

    @torch.jit.script_method
    def forward(self, x):
        return self.conv(x)

M().save("m.pt")
loaded = torch.jit.load("m.pt")
loaded(torch.ones(20, 20))