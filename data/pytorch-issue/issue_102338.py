import torch.nn as nn

import torch
import torch._dynamo
import logging
import torch.nn.functional as F

torch._logging.set_logs(dynamo=logging.DEBUG)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


x = torch.randn(10, 10)

m = MyModule()
print(m(x))

opt_m = torch.compile(backend="eager")(m)
print(opt_m(x))