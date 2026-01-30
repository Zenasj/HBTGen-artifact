import torch.nn as nn

import torch

print(torch.__version__)

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.zeros(100)
        
print(M().state_dict().keys())

print(torch.jit.script(M()).state_dict().keys())

class M2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('a', torch.zeros(100), persistent=False)
        
print(M2().state_dict().keys())

print(torch.jit.script(M2()).state_dict().keys())