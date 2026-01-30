# Run with torchrun and make sure you have CUDA device available:
# > torchrun issue.py

import torch
import torch.nn as nn
import torch.distributed

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)
    
    def forward(self, x):
        # return self.linear(x) # uncomment this line and comment the "forward" line below and everything works
        return self.linear.forward(x)

torch.distributed.init_process_group(torch.distributed.Backend.NCCL)

model = Model()
model = model.to('cuda')
model = nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0)
model = torch.compile(model)
print(model(torch.randn(32, device='cuda')))

torch.distributed.destroy_process_group()

import torch
import torch.nn as nn
import torch.distributed

torch._dynamo.config.inline_inbuilt_nn_modules = True