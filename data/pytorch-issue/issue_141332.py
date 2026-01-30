import torch.nn as nn

python
import os
import torch
from torch._subclasses import FakeTensorMode, CrossRefFakeMode
import torch.nn.functional as F

fake_mode = FakeTensorMode()
cross_ref_mode = CrossRefFakeMode()


with cross_ref_mode:
    input = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0],
device='xpu')
    F.logsigmoid(input)