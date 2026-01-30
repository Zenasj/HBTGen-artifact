import torch.nn as nn

import torch
import torch.nn.functional as F

torch.manual_seed(1234)
tin = torch.rand((1, 512, 1245), dtype=torch.float32)
tparams = torch.rand((512, 256, 16), dtype=torch.float32)
tbias = torch.rand((256), dtype=torch.float32)

device = 'cpu'
tcpu = F.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

device = 'mps'
tgpu = F.conv_transpose1d(tin.to(device), tparams.to(device), tbias.to(device), stride=8, padding=4)

torch.all(torch.isclose(tcpu, tgpu.cpu()))