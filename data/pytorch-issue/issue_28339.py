import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict(
            {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
        )
        self.activations = nn.ModuleDict(
            [["lrelu", nn.LeakyReLU()], ["prelu", nn.PReLU()]]
        )

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

torch.jit.script(MyModule())