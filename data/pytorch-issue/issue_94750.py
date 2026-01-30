import torch.nn as nn

import torch
import torch._dynamo

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dims = [3, 1, 0, 2]

    def forward(self, x):
        res = x
        sorted_dims = sorted(self.dims)
        for i in range(len(self.dims)-1, -1, -1):
            res = torch.squeeze(res, dim=sorted_dims[i])
        return res

x = torch.rand((2, 3, 4, 5))
model = MyModule()
opt_model = torch._dynamo.optimize("eager")(model)
print(opt_model(x))