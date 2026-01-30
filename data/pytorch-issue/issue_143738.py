import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
from torch._inductor import config

config.fallback_random = True


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        torch.manual_seed(0)
        x = torch.argsort(x, dim=3)
        # x.dtype: torch.int64
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


model = Model()


x = torch.randn(1, 1, 2, 4)

inputs = [x]

output = model(*inputs)

c_model = torch.compile(model)
c_output = c_model(*inputs)

print(output)
print(c_output)