import torch.nn as nn

import torch
from einops import rearrange
    

class MyAutocastModel(torch.nn.Module):
    def forward(self,x):
        with torch.autocast(device_type="cuda"):
            return rearrange(x,"B C H W -> B H W C")
        
model = torch.compile(MyAutocastModel())
model(torch.randn(3,3,32,32).cuda())