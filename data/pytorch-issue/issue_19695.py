import torch.nn as nn

import torch
a = torch.nn.Embedding(3, 4, sparse=True).half().cuda()
a(torch.LongTensor([1, 0]).cuda()).sum().backward()