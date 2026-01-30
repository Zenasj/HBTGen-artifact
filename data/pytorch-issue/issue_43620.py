import torch
import torch.nn as nn

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = torch.nn.Conv2d(3, 32, 1, 1)
     
    def forward(self, x):
        x = self.layer1(x)
        return x

model = MyModel().eval().cuda()

shape = [1, 3, 32, 32]
stride = (3072, 1, 96, 3)
dummy_input = torch.randn(shape).as_strided(shape, strides).cuda()
traced_module = torch.jit.trace(model, dummy_input)
freeze_module = torch._C._freeze_module(traced_module._c)
freeze_module.forward(dummy_input)