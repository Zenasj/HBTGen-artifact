import torch.nn as nn

import torch
import torch._dynamo
import logging
import warnings

torch._logging.set_logs(dynamo=logging.DEBUG)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.layer(x)


x = torch.randn(10, 10)

m = MyModule()
print(m(x))

opt_m = torch.compile(backend="eager", fullgraph=True)(m)
print(opt_m(x))