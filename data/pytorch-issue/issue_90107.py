import torch.nn as nn

import torch

def gen_uniform(shape):
    r = torch.empty(shape, dtype=torch.float32)
    return r.uniform_(0, 1)

class Model1(torch.nn.Module):
    def forward(self):
        return gen_uniform([10, 20])

model1 = torch.compile(Model1())
print(model1())