py
import torch
import torch.nn as nn
import logging

class CompiledClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.nums = torch.tensor([1,2,3,4,5,6,7,8,9,10])
        self.t = 5
    
    def forward(self):
        self.num = self.nums[self.t//12]
        self.t += 1
        return self.num
    
m = CompiledClass()
m = torch.compile(m, backend="eager")

torch._logging.set_logs(dynamo = logging.DEBUG)
torch._dynamo.config.verbose = True

# the first call works
m()
# the second call causes a failure
m()