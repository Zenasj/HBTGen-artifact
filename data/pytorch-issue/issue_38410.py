import torch.nn as nn

import torch
from torch.backends import cudnn
torch.manual_seed(0)

cudnn.deterministic = True
cudnn.benchmark = False

import torch
from torch.backends import cudnn
torch.manual_seed(0)

cudnn.deterministic = True # type: ignore
cudnn.benchmark = False # type: ignore