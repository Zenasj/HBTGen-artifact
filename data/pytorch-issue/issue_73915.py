import torch.nn as nn

import torch

a = torch.zeros([1], dtype=torch.complex128)
a[0] = -3.2427e-04+5.8708e-03j
b = torch.zeros([0], dtype=torch.complex128)
print(a)
print(torch.clamp_min(a, b))

a = torch.randn(4, dtype=torch.cfloat)
angle_ = a.angle()
clamped_abs = torch.clamp(a.abs(), min=...)
b = torch.polar(clamped_abs, angle_)

class Feedforward(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(Feedforward, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size).to(torch.complex128)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1).to(torch.complex128)
            self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden.real) + 1j * self.relu(hidden.imag)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output