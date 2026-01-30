import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, v1):
        v2 = torch.add(v1, 3.589, alpha=57) # ERROR!
        # v2 = torch.add(v1, v1, alpha=57) # works fine
        # v2 = v1 + 3.589820384979248 + 57 # works fine
        return v2

model = Model()
x = torch.rand(1024)
compiled = torch.compile(model, fullgraph=True)
print(compiled(x))