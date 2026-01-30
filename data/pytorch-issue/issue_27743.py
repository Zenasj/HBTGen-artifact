import torch
torch.jit.trace(lambda x: {'out': x}, torch.arange(3))

import torch
torch.jit.trace(lambda x: x['input'], {'input': torch.arange(3)})