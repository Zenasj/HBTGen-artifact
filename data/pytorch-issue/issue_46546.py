import torch
x = torch.rand(1, 5, dtype=torch.cfloat)
torch.matmul(x, x.T)

import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(ComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.complex64))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

layer = ComplexLinear(5, 5)
x = torch.rand(1, 5, dtype=torch.cfloat)
layer(x)

def mse_loss(input, target):
  return (input - target).abs().square().mean()

model = ComplexLinear(5, 5).cuda()
optimizer = torch.optim.Adam(model.parameters())
x = torch.rand(10, 5, dtype=torch.complex64).cuda()
x_out = model(x)
loss = F.mse_loss(x_out, x)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(float(loss))