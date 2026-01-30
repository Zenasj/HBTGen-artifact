import torch.nn as nn

import torch
import torch._dynamo
import logging
import torch.nn.functional as F
import numpy as np

torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, inductor=logging.DEBUG)

class MyModule(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.temperature = 1
        self.layer = torch.nn.Softmax(dim=1)

    def forward(self, x):
        n_samples, _ = x.shape
        y = 1.0 * torch.ones(n_samples, dtype=x.dtype, device=x.device)
        inp = x / y[..., None]
        return self.layer(inp)


x = torch.rand([4, 4])

m = MyModule()
print(m(x))

opt_m = torch.compile(backend="inductor")(m)
print(opt_m(x))