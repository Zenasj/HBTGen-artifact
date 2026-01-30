import torch.nn as nn

import torch
from torch import nn

mod = nn.Linear(3, 4)
state_meta = dict()
for k, v in mod.state_dict().items():
  state_meta[k] = v.to('meta')

a.load_state_dict(state_meta)

mod.weight = nn.Parameter(mod.weight.to('meta'))

import torch
from torch import nn

mod = nn.Linear(3, 4)
state_meta = dict()
for k, v in mod.state_dict().items():
  state_meta[k] = v.to('meta')

a = nn.Linear(3, 4)
a.load_state_dict(state_meta, assign=True)  # runs without error
print(a.state_dict())
# OrderedDict([('weight', tensor(..., device='meta', size=(4, 3))), ('bias', tensor(..., device='meta', size=(4,)))])