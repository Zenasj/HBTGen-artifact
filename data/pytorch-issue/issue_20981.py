import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import nn

m = nn.Conv2d(1, 1, 1, 1, padding=1, padding_mode='circular')
x = torch.rand(1, 1, 5, 5)

print(m(x).shape)

expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

from torch import nn

x = torch.rand(1, 16, 32, 32)

filter_pzeros = nn.Conv2d(16, 16, 3, padding=1, padding_mode='zeros')
filter_pcircular = nn.Conv2d(16, 16, 3, padding=1, padding_mode='circular')

print(filter_pzeros(x).shape)
print(filter_pcircular(x).shape)

torch.Size([1, 16, 32, 32])
torch.Size([1, 16, 30, 30])

expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

from torch.nn import functional as F

x = torch.rand(1, 16, 32, 32)

y_pconstant = F.pad(x, (1, 0, 1, 0), mode='constant')
y_preplicate = F.pad(x, (1, 0, 1, 0), mode='replicate')
y_preflect = F.pad(x, (1, 0, 1, 0), mode='reflect')
y_pcircular = F.pad(x, (1, 0, 1, 0), mode='circular')

print(y_pconstant.shape)
print(y_preplicate.shape)
print(y_preflect.shape)
print(y_pcircular.shape)

torch.Size([1, 16, 33, 33])
torch.Size([1, 16, 33, 33])
torch.Size([1, 16, 33, 33])
torch.Size([1, 16, 32, 32])