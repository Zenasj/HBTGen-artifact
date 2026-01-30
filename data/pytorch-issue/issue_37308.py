import torch
import torch.nn as nn

class Model(nn.Module):
  
  def __init__(self):
    super(Model, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=0)
    self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
    
    self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
    self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
    
    self.clasfr = nn.Conv2d(2592, 10, 1)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    
    x = self.conv3(x)
    x = self.conv4(x)
    
    x = x.view(128, -1, 1, 1)
    x = self.clasfr(x)
    
    return x
    
model = Model()

batch = torch.rand(128, 1, 28, 28)

out = model(batch)

with torch.autograd.profiler.profile(use_cuda=False) as prof:
  out.sum().backward()
  
print(prof.key_averages().table(sort_by="self_cpu_time_total"))