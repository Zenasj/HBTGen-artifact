import torch.nn as nn

import torch
upsampling = torch.nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear') # or any mode you like