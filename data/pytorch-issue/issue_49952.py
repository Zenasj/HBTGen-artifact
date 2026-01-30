import torch
print(torch.cuda.memory_allocated(0))

import torch
#print(torch.cuda.get_device_name(0))
# or
torch.cuda.init()
print(torch.cuda.memory_allocated(0))