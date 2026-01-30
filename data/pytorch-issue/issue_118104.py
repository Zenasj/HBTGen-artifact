python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, other, out):
        input = torch.logical_xor(input=input, other=other, out=out)
        return input

x = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32)
y = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32)
z = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float16)

model = Model().to(torch.device('cpu'))
eag = model(x, y, z)
opt = torch.compile(model.forward, mode='reduce-overhead')(x, y, z)