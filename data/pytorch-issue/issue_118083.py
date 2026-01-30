import torch.nn as nn

import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, out, input, other):  
        out = torch.ge(out=out, input=input,other=other)        
        return out

input1 = torch.rand([8, 6, 8, 1, 6, 1, 3, 4, 1, 3], dtype=torch.float16)
input2 = torch.rand([8, 6, 8, 1, 6, 1, 3, 4, 1, 3], dtype=torch.float32)
input3 = torch.rand([8, 6, 8, 1, 6, 1, 3, 4, 1, 3], dtype=torch.float32)

model = Model().to(torch.device('cpu'))
opt = torch.compile(model.forward, mode='max-autotune')(input1, input2, input3)