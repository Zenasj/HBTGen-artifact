import torch
import torch.nn as nn

from torch._inductor import config

config.fallback_random = True


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.FractionalMaxPool2d(kernel_size=(1, 1), output_ratio=(0.5, 0.5))

    def forward(self, x):
        x = self.pool(x)
        return x


model = Model().eval()
c_model = torch.compile(model)  # backend="cudagraph" is OK

x = torch.randn(1, 1, 10, 10)  # dtype=torch.float64 also triggers the inconsistency
inputs = [x]

torch.manual_seed(0)
output1 = model(*inputs)

torch.manual_seed(0)
output2 = model(*inputs)

torch.manual_seed(0)
c_output = c_model(*inputs)

print(torch.allclose(output1, output2))
print(torch.allclose(output1, c_output))
print(torch.max(torch.abs(output1 - c_output)))

import torch
import torch.nn as nn

from torch._inductor import config

config.fallback_random = True


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool = nn.FractionalMaxPool2d(kernel_size=(1, 1), output_ratio=(0.5, 0.5))

    def forward(self, x):
        x = self.pool(x)
        return x


model = Model().eval()
c_model = torch.compile(model)  # backend="cudagraph" is OK

x = torch.randn(1, 1, 10, 10)  # dtype=torch.float64 also triggers the inconsistency
inputs = [x]

torch.manual_seed(0)
output1 = model(*inputs)

torch.manual_seed(0)
output2 = model(*inputs)

torch.manual_seed(0)
c_output = c_model(*inputs)

print(torch.allclose(output1, output2))  # True
print(torch.allclose(output1, c_output))  # False
print(torch.max(torch.abs(output1 - c_output)))  # tensor(2.7663)