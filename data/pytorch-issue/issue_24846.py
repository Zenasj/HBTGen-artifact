import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as dist
import math

mean = Parameter(torch.Tensor(1, 2))
log_std = Parameter(torch.Tensor(1, 2))

n = dist.Normal(mean, torch.exp(log_std))

nn.init.kaiming_uniform_(mean, a=math.sqrt(5))
nn.init.normal_(log_std, -5)

print(n.scale)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as dist
import math

mean = Parameter(torch.Tensor(1, 2))
log_std = Parameter(torch.Tensor(1, 2))

nn.init.kaiming_uniform_(mean, a=math.sqrt(5))
nn.init.normal_(log_std, -5)

n = dist.Normal(mean, torch.exp(log_std))

print(n.scale)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as dist
import math

mean = Parameter(torch.tensor(np.zeros((1, 2)), dtype=torch.float32))
log_std = Parameter(torch.tensor(np.zeros((1, 2)), dtype=torch.float32))

n = dist.Normal(mean, torch.exp(log_std))

nn.init.kaiming_uniform_(mean, a=math.sqrt(5))
nn.init.normal_(log_std, -5)

print(n.scale)

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.distributions as dist
import math

mean = Parameter(torch.Tensor(1, 2))
log_std = Parameter(torch.Tensor(1, 2))

nn.init.kaiming_uniform_(mean, a=math.sqrt(5))
nn.init.normal_(log_std, -5)

n = dist.Normal(mean, torch.exp(log_std))

print('Before reinit:')
print(torch.exp(log_std))
print(n.scale)
print(mean)
print(n.loc)

nn.init.normal_(mean, 5)
nn.init.normal_(log_std, 5)

print('After reinit:')
print(torch.exp(log_std))
print(n.scale)
print(mean)
print(n.loc)