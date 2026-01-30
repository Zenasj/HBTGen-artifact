import torch.nn as nn

input_size = 5
output_size = 2
self.fc = nn.Linear(input_size, output_size)

import torch
torch.cuda.init()
x = torch.rand(1000, 1000)
y = x.to(0) # got ~ 1s delay here

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False