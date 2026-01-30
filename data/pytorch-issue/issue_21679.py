import torch
import torch.nn as nn

m = nn.BatchNorm2d(2)
m.weight.data.fill_(1)
m.bias.data.fill_(3)
m.running_mean.data.fill_(2)
m.running_var.data.fill_(4)

test_input = torch.ones(1, 2, 2, 2)

print(m(test_input))
print(m(test_input))