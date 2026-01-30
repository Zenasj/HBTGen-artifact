import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

n = Net()
example_forward_input = torch.rand(1, 1, 3, 3)


# Trace a module (implicitly traces `forward`) and construct a
# `ScriptModule` with a single `forward` method
module = torch.jit.trace(n, example_forward_input)


class ScriptNet(nn.Module):
    def __init__(self):
        super(ScriptNet, self).__init__()
        self.net = torch.jit.trace(n, example_forward_input)
    def forward(self, input):
        return self.net(input)

scripted_mod = torch.jit.script(ScriptNet())
print(scripted_mod.graph)
print(scripted_mod(example_forward_input))
scripted_traced = torch.jit.script(module)
scripted_traced(example_forward_input)