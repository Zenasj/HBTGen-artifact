import torch.nn as nn

import torch

class HaveSequential(torch.jit.ScriptModule):
    __constants__ = ['someseq']

    def __init__(self):
        super(HaveSequential, self).__init__()
        self.someseq = torch.nn.Sequential(
                            torch.nn.Conv2d(1, 20, 5),
                            torch.nn.ReLU(),
                            torch.nn.Conv2d(20, 64, 5),
                            torch.nn.ReLU()
                        )

    @torch.jit.script_method
    def forward(self, x):
        return self.someseq(x)

hs = HaveSequential()
hs.save("test.ptn")