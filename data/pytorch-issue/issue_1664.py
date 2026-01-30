import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.backends.cudnn as cudnn
cudnn.enabled = True
cudnn.benchmark = False

x = Variable(torch.ones(1,32,192,256,512)).cuda()

model = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1, bias=False).cuda()

y = model(x)
print(y.size())