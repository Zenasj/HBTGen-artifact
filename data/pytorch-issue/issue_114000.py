import torch.nn as nn

py
import torch
import unittest
from torch.export import export

class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            conv = self.conv(x)
            relu = self.relu(conv)
            mul = relu*0.5
            return mul

input = torch.randn((1, 3, 224, 224), dtype=torch.float).to("cuda")
model = MyModule().eval().cuda()

exp_program = export(model, tuple([input]))

torch.export.save(exp_program, "./trt.ep")
deserialized_prog = torch.export.load("./trt.ep")
out_pyt = model(input)
out_ser = deserialized_prog(input).cuda()