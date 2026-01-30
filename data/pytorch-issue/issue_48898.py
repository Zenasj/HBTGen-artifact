import torch.nn as nn

import torch

def linear_inplace(layer, v):
    return torch.addmm(layer.bias, v, layer.weight.t(), out=v)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(16, 16)
        ])

        for i in range(len(self.layers)):
            self.layers[i] = torch.nn.utils.weight_norm(self.layers[i])

    def forward(self, x):
        layer = self.layers[0]
        return layer(x)
        # return linear_inplace(layer, x)

device = torch.device('cuda')

m = Model().to(device)
input = torch.ones(5, 16, device=device)

with torch.no_grad():
    print(m(input))