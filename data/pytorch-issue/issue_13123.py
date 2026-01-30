import dotenv
import django
dotenv.load_dotenv()
django.setup()

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


inputs_tensor = torch.rand(10, 4)
targets_tensor = torch.rand(10, 1)

net = Network()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

optimizer.zero_grad()
outputs = net(inputs_tensor)
loss = criterion(outputs, targets_tensor)

loss.backward()
optimizer.step()