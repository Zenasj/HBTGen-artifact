import torch.nn as nn

import torch

@torch.jit.script
def pre_dct_2d(x, N: int):
    B, C, H, W = x.shape
    x = x.view(B, C, H // N, N, W // N, N)
    x = x.permute(0, 1, 2, 4, 3, 5)  # B, C, H//S, W//S, S, S
    return x

def post_dct_2d(x):
    x = x.permute(0, 1, 2, 4, 3, 5)
    x = x.flatten(2,3).flatten(3,4)
    return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = pre_dct_2d(x,8)
        x = post_dct_2d(x)
        return x

if __name__ == '__main__':
    n = Model()
    n_jit = torch.jit.script(n)

    x = torch.randn(1,3,8,8)
    y1 = n(x)
    y2 = n_jit(x)

    torch.onnx.export(n, (x,), "poop.onnx", opset_version=12)

torch.jit.script

post_dct_2d(x)

x = x.view(B, C, H // N, N, W // N, N)