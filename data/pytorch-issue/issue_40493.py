import torch.nn as nn

import torch


class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv_1 = torch.nn.Conv2d(torch.nn.parameter.ParameterMode.Infer, 4, 2)
        self.conv_2 = torch.nn.Conv2d(torch.nn.parameter.ParameterMode.Infer, 4, 2)
        self.linear = torch.nn.Linear(torch.nn.parameter.ParameterMode.Infer, 10)

    def forward(self, x):
        y = self.conv_1(x).clamp(min=0)
        z = self.conv_2(y).clamp(min=0)
        return self.linear(z)


net = MyNetwork()
net.infer_parameters(torch.ones(5, 5, 5, 10))
print(net.conv_1.weight.shape)
print(net.conv_2.weight.shape)
print(net.linear.weight.shape)

Linear(ParameterMode.Infer, 10)

Linear(5, 10, defer_parameters=True)

lin1 = nn.Linear(infer, 5)
lin2 = nn.Linear(infer, 5)
lin2.weight = lin1.weight
lin1.infer_parameters(torch.ones(5, 10))
# now lin2 has the same (initialized) weight