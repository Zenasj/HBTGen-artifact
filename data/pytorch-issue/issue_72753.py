import torch
import torch.nn as nn

from torch.backends import cudnn
cudnn.allow_tf32 = False  # error: Module has no attribute "allow_tf32"