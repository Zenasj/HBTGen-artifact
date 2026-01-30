import torch.nn as nn

py
import torch

scripted = torch.jit.script(torch.nn.Conv2d(3, 3, 3))
img = torch.rand(1, 3, 20, 20)

import warnings
warnings.simplefilter('module')

scripted(img)