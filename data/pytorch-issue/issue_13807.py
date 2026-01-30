import torch.nn as nn

import torch
import torch.backends.cudnn

torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor([1.]))
# If it returns True, then cudnn is enabled.