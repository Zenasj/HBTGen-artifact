import torch.nn as nn

import torch

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, param):
        z = torch.sum(**param)
        return z


if __name__ == "__main__":
    model = Repro()
    params = {'dim': [1, -1], 'keepdim': True, "input" : torch.randn([64, 54, 43]).to(torch.int32)}
    params["out"] = torch.empty(0, dtype=torch.int32)
    model = torch.compile(model, backend="eager")
    res = model(params)
    print(res)

import torch

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, param):
        z = torch.sum(**param)
        return z


if __name__ == "__main__":
    model = Repro()
    params = {'dim': [1, -1], 'keepdim': True, "input" : torch.randn([64, 54, 43]).to(torch.int32)}
    params["out"] = torch.empty(0, dtype=torch.int32)
    model = torch.compile(model, backend="eager")
    res = model(params)
    print(res)