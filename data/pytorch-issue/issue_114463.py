import torch.nn as nn

import torch
batch_size=191
kernel_size=1
data = torch.randn([batch_size, 3072, 299], dtype=torch.float, requires_grad=True).to("cuda:0")
net = torch.nn.Conv1d(3072, 3072, kernel_size=kernel_size, padding=0, stride=1, dilation=1, groups=1).to("cuda:0")
out = net(data)
out.backward(torch.randn_like(out))
torch.cuda.synchronize()