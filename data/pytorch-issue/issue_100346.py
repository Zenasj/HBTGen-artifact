import torch.nn as nn

import torch
import torch._dynamo


class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        var = torch.randn(2, 4, requires_grad=True).mean()
        return var


mod = Repro()
opt_mod = torch._dynamo.optimize("inductor")(mod)


loss = mod()
loss.backward()

loss = opt_mod() 
loss.backward() # or print(loss.requires_grad) here which returns False