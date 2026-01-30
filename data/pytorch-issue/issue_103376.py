import torch.nn as nn

import torch
a = torch.nn.Parameter()
a *= 2
reveal_type(a.mul_(2))
print(a.mul_(2).__class__)  # torch.nn.parameter.Parameter