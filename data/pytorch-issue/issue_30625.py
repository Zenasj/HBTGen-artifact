import torch
import torch.nn as nn
import numpy as np
import random

class MyModule(nn.Module):
    def forward(self):
        pass

    @torch.jit.ignore
    def call_np(self) -> int:
        return np.random.choice(2, p=[.95,.05])

    @torch.jit.export
    def func(self):
        done = self.call_np()
        print (done)

scripted_module = torch.jit.script(MyModule())
print(scripted_module.func.graph)
scripted_module.func()