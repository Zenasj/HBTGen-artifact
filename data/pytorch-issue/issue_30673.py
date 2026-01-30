import torch
import torch.nn as nn
import numpy as np
import random

@torch.jit.ignore
def call_np():
    return torch.jit.export(np.random.choice(2, p=[.95,.05]))

def forward(self):
    pass 

@torch.jit.export
def func(self):
    done = self.call_np()
    print (done)
scripted_module = torch.jit.script(MyModule())
scripted_module.func()

class MyModule(nn.Module):
    @torch.jit.ignore
    def call_np(self):
        # type: () -> int
        return np.random.choice(2, p=[.95,.05])

    def forward(self):
        pass

    @torch.jit.export
    def func(self):
        done = self.call_np()
        print (done)

scripted_module = torch.jit.script(MyModule())
scripted_module.func()