import torch
import torch.nn as nn

import torc
device=torch.device('cuda:6') # use any device other than default cuda:0
net = torch.nn.Linear(100, 100) 
net_cuda = net.to(device)  
print(torch.cuda.memory_allocated(device)) # prints 0
torch.cuda.set_device(device) 
print(torch.cuda.memory_allocated(device)) # prints 40960