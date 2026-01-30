import torch
import torch.nn as nn

print(torch.__version__)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.modules_1 = nn.ModuleList([nn.Identity()])
        self.modules_2 = nn.ModuleList([nn.Identity()])

    def forward(self, x):
        rv = []
        for mod1, mod2 in zip(self.modules_1, self.modules_2, strict=True):
            rv.append((mod1(x), mod2(x)))
        return rv

net = Net()
net = net.cuda()
net = torch.compile(net, backend="eager", fullgraph=True)
out = net(torch.Tensor([1, 2, 3, 4]))
print(out)