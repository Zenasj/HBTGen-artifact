import torch.nn as nn

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)
device = torch.device('cuda')
print(torch.cuda.get_device_properties(device))
print(torch.tensor([1.0, 2.0]).cuda())