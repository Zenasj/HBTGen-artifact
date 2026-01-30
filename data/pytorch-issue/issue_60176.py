3
import torch
import torch.nn as nn
patches = torch.rand(2,1,32,32)
m = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
with torch.no_grad():
    output = m(patches)
print (output[0,0:10,0,0])
output2 = m(patches)
print (output2[0,0:10,0,0])