import torch
import torch.nn as nn

with torch.backends.cudnn.flags(deterministic=False):  #  <--- wrong
    # the following code will not use cudnn/miopen
    pass