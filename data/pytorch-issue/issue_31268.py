import torch.nn as nn

import torch
from torchvision import models

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.submodule = models.resnet50()
    def forward(self, x):
        return self.submodule(x)

my_module = MyModule()
my_module.eval()
x = torch.rand(1, 3, 224, 224)
traced_cell = torch.jit.trace(my_module, x)
traced_cell.save('my_module.pth')
loaded_model = torch.jit.load('my_module.pth')
loaded_model.copy()