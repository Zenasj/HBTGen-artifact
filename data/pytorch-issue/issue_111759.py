import torch.nn as nn

import torch
from functorch.compile import aot_function

model = torch.nn.Linear(20, 30)
x = torch.randn(128, 20)

def backend(gm, inputs):
    return gm

model = aot_function(model, fw_compiler=backend, bw_compiler=backend)
model(x)