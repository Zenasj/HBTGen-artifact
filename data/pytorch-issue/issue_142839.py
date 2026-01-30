import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.inn = nn.InstanceNorm2d(num_features=3)

    def forward(self, x):
        x = self.inn(x)
        return x


model = Model()

x = torch.randn(1, 3, 1024, 1024)  # As `H` and `W` increase, the error will be amplified

inputs = [x]

c_model = torch.compile(model)
output = model(*inputs)
c_output = c_model(*inputs)

print(torch.allclose(output, c_output, 1.3e-6, 1e-5))  # I set a less strict value
print(torch.max(torch.abs(output - c_output)))

model = Model().cuda()

x = torch.randn(1, 3, 11024, 11024).cuda()