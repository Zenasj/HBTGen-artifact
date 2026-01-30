import torch.nn as nn

import torch

print(torch.__version__)
print(torch.version.git_version)

l = torch.nn.Linear(100, 1)
print(l(torch.randn(0, 100)))