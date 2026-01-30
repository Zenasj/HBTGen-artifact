import torch.nn as nn

import torch
from torch import nn
import numpy as np

class Dummy:
    def __init__(self, x):
        self.x = x

def nemo(x):
    return x+1


class Demo(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print(torch.jit.is_tracing())
        if torch.jit.is_tracing():
            return x
        else:
            return nemo(x) # Dummy(x)


if __name__ == "__main__":
    input_tensor = torch.range(20, 80)
    demo = Demo()
    out = demo(input_tensor)
    traced_model = torch.jit.trace(demo, input_tensor, strict=False)