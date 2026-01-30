import torch
import torch.nn as nn

batch_size=10
layer = torch.nn.Linear(3000,512).cuda()
x = torch.rand(batch_size,5000).cuda()
y = layer(x)
print(y.shape) #torch.Size([10, 512])

batch_size=10
layer = torch.nn.Linear(3000,512).cuda()
x = torch.rand(batch_size,5000).cuda()
y = layer(x[:,:100])
print(y.shape) #torch.Size([10, 512])