import torch
from torch.autograd import Variable

a = Variable(torch.FloatTensor([1]))
a.set_(a.storage(), storage_offset=0, size=a.size(), stride=a.stride())

a = torch.FloatTensor([1])
a.set_(a.storage(), storage_offset=0, size=a.size(), stride=a.stride())

import torch
from torch.autograd import Variable

a = Variable(torch.FloatTensor([1]))
a.data.set_(a.data.storage(), storage_offset=0, size=some_size, stride=some_stride)