import torch.nn as nn

py
import torch

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)

    def forward(self, x1, split_sections, dim):
        t1 = torch.split(x1, split_sections, dim)
        targets = []
        for t in t1:
            targets.append(t)
        for target in targets:
            t2 = torch.stack(targets, dim)
            t3 = torch.tanh(t2)
        return t3


func = Model().to('cuda').eval()

x = torch.randn(1, 3, 64, 64).cuda()
split_sections = 1
dim = 1

with torch.no_grad():
    print(func(x.clone(), split_sections, dim))

    func1 = torch.compile(func)
    print(func1(x.clone(), split_sections, dim))