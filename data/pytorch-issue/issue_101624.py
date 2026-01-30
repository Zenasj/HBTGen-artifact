import torch
import logging
import torch._dynamo

torch._logging.set_logs(dynamo=logging.DEBUG, bytecode=True)

import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(420)

class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = torch.nn.Linear(5, 5)

    def forward(self, x):
        x = x + 1
        with torch.cuda.amp.autocast(dtype=torch.float16):
            x = self.linear(x)
            x = torch.sin(x)
        x = torch.cos(x)
        x = x - 1
        return x

x = torch.randn(5, 5, device="cuda")

m = MyModel().to("cuda")
print(m(x))

opt_m = torch.compile(backend="eager")(m)
print(opt_m(x))

l__self___linear = self.L__self___linear(add)

sin = torch.sin(l__self___linear);

torch.with_grad

torch.amp.autocast