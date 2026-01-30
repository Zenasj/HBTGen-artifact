import torch.nn as nn

python
import torch
class Net(torch.nn.Module):
    def forward(self, x):
        for i in range(1000):
            x = 1.0 * x
        return x
net = Net()
net = torch.compile(net)
x = torch.tensor([1.0])
print(net(x))

while (cont := cont()) is not None:
    pass