import torch.nn as nn

import torch

conv = torch.nn.Conv2d(1, 16, 3, stride=1)
for c in range(100):
	x = torch.randn(1, 80, 140)
	x = x.unsqueeze(1)
	x = conv(x)
	if torch.isnan(torch.sum(x)):
		print(c)