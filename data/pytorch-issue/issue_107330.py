import torch
torch._C._cuda_attach_out_of_memory_observer(fn)

import torch
torch.cuda.init()
torch._C._cuda_attach_out_of_memory_observer(fn)